# app.py
import streamlit as st
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.query import MetadataQuery, Filter
import pickle
import numpy as np
from scipy.sparse import load_npz
import pandas as pd
import spacy
from scispacy.abbreviation import AbbreviationDetector
from tqdm import tqdm
import requests
import sys
import os


# ----------------------------
# Query to ClinicalTrials.gov
# ----------------------------
def query_clinical_trials(condition, intervention, max_results=10):
    """
    Query the ClinicalTrials.gov API to retrieve studies based on Condition and Intervention.
    """
    # API Endpoint
    api_url = "https://clinicaltrials.gov/api/v2/studies"

    # Parameters
    params = {
        'query.cond': condition,
        'query.intr': intervention
    }

    try:
        response = requests.get(api_url, params=params)
        response.raise_for_status()  # Raise an error for bad status codes
    except requests.exceptions.RequestException as e:
        st.error(f"Error querying ClinicalTrials.gov API: {e}")
        return pd.DataFrame()  # Return empty DataFrame

    data = response.json()

    # Check if 'studies' key exists
    if 'studies' not in data:
        st.warning("No 'studies' key found in the API response. No studies found.")
        return pd.DataFrame()

    studies = data['studies']
    if not studies:
        st.warning("No studies matched your query.")
        return pd.DataFrame()

    # Extract NCT IDs and Study Titles
    nct_ids = []
    study_titles = []
    for study in studies:
        protocol_section = study.get('protocolSection', {})
        identification_module = protocol_section.get('identificationModule', {})
        nct_id = identification_module.get('nctId', 'N/A')
        brief_title = identification_module.get('briefTitle', 'No Title Provided')
        nct_ids.append(nct_id)
        study_titles.append(brief_title)

    results_df = pd.DataFrame({
        'nct_id': nct_ids,
        'Study Title': study_titles
    })

    # Limit to max_results if needed (though the param doesn't directly limit in the API)
    return results_df.head(max_results)


# ----------------------------
# Display Trials
# ----------------------------
def display_trials(trials_df):
    """
    Display the trials in a Streamlit table.
    """
    st.subheader("Top Trials from ClinicalTrials.gov")
    st.dataframe(trials_df)


# ----------------------------
# Weaviate Collection Info
# ----------------------------
COLLECTIONS = {
    'title': {
        'name': 'TrialTitle',
        'weight': 0.25
    },
    'primary': {
        'name': 'TrialPrimaryOutcome',
        'weight': 0.45
    },
    'secondary': {
        'name': 'TrialSecondaryOutcome',
        'weight': 0.30
    },
    'inclusion': {
        'name': 'TrialInclusion',
        'weight': 0.15
    },
    'exclusion': {
        'name': 'TrialExclusion',
        'weight': 0.15
    },
    'conditions': {
        'name': 'TrialConditions',
        'weight': 0.35
    },
    'interventions': {
        'name': 'TrialInterventions',
        'weight': 0.35
    }
}


# ----------------------------
# Entity Reranker
# ----------------------------
class EntityReranker:
    def __init__(self):
        st.write("Loading SciSpacy NER model (BC5CDR)...")
        self.nlp = spacy.load("en_ner_bc5cdr_md")
        self.nlp.add_pipe("abbreviation_detector", first=True)

    def extract_trial_entities(self, trial_data):
        entities = {
            'drugs': set(),
            'diseases': set(),
            'abbreviations': {}
        }

        # Process each relevant field
        for field in ["study_title", "conditions", "interventions", "criteria"]:
            if field not in trial_data:
                continue

            text = str(trial_data[field])
            doc = self.nlp(text)

            # Get abbreviations
            for abrv in doc._.abbreviations:
                entities['abbreviations'][str(abrv)] = str(abrv._.long_form)

            # Get entities
            for ent in doc.ents:
                if ent.label_ == 'CHEMICAL':
                    entities['drugs'].add(ent.text.lower())
                    if ent.text in entities['abbreviations']:
                        entities['drugs'].add(entities['abbreviations'][ent.text].lower())
                elif ent.label_ == 'DISEASE':
                    entities['diseases'].add(ent.text.lower())
                    if ent.text in entities['abbreviations']:
                        entities['diseases'].add(entities['abbreviations'][ent.text].lower())

        return entities

    def compute_entity_similarity(self, entities1, entities2):
        drugs1, drugs2 = entities1['drugs'], entities2['drugs']
        diseases1, diseases2 = entities1['diseases'], entities2['diseases']

        drug_sim = len(drugs1 & drugs2) / max(len(drugs1 | drugs2), 1)
        disease_sim = len(diseases1 & diseases2) / max(len(diseases1 | diseases2), 1)

        return (drug_sim + disease_sim) / 2.0


# ----------------------------
# Multi-stage Similarity
# ----------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))

def find_similar_trials(query_nct_id, client,
                        n_stage1=40, n_final=10,
                        cat_features_path=os.path.join(base_dir, 'data', 'categorical_features.npz'),
                        cat_index_path=os.path.join(base_dir, 'data', 'categorical_index.pkl')):
    """Multi-stage similarity search using direct filters."""

    # Load categorical features
    try:
        cat_features = load_npz(cat_features_path)
    except FileNotFoundError:
        st.error(f"Categorical features file not found at: {cat_features_path}")
        return []

    # Load the categorical index
    try:
        with open(cat_index_path, 'rb') as f:
            cat_index = pickle.load(f)
    except FileNotFoundError:
        st.error(f"Categorical index file not found at: {cat_index_path}")
        return []

    # Stage 1: Outcome-based similarity
    st.write("### Stage 1: Outcome-based similarity (title, primary, secondary)")
    stage1_results = {}

    stage1_collections = ['title', 'primary', 'secondary']
    for coll_type in stage1_collections:
        collection = client.collections.get(COLLECTIONS[coll_type]['name'])
        weight = COLLECTIONS[coll_type]['weight']

        # Get query vector using filter
        response = collection.query.fetch_objects(
            filters=Filter.by_property("nct_id").equal(query_nct_id),
            include_vector=True,
            limit=1
        )

        if not response.objects:
            st.warning(f"Query trial not found in {coll_type}")
            continue

        # Get query vector and find similar trials
        query_vector = response.objects[0].vector['default']
        similar = collection.query.near_vector(
            near_vector=query_vector,
            limit=n_stage1 + 1,
            return_metadata=MetadataQuery(distance=True)
        )

        # Store results
        for obj in similar.objects:
            if obj.properties['nct_id'] == query_nct_id:
                continue

            nct_id = obj.properties['nct_id']
            if nct_id not in stage1_results:
                stage1_results[nct_id] = {
                    'properties': obj.properties,
                    'score': 0,
                    'similarities': {}
                }

            similarity = 1 - obj.metadata.distance
            stage1_results[nct_id]['score'] += similarity * weight
            stage1_results[nct_id]['similarities'][coll_type] = similarity

    # Get top trials from Stage 1
    stage1_trials = sorted(
        stage1_results.items(),
        key=lambda x: x[1]['score'],
        reverse=True
    )[:n_stage1]

    st.write(f"Stage 1: Selected top {len(stage1_trials)} trials.")

    # Stage 2: Criteria-based similarity
    st.write("### Stage 2: Criteria-based similarity")
    stage2_results = {}

    # Initialize with Stage 1 data
    for nct_id, trial_data in stage1_trials:
        stage2_results[nct_id] = {
            'properties': trial_data['properties'],
            'stage1_score': trial_data['score'],
            'stage2_score': 0,
            'similarities': trial_data['similarities']
        }

    # Process each Stage 2 collection
    stage2_collections = ['inclusion', 'exclusion', 'conditions', 'interventions']
    for coll_type in stage2_collections:
        collection = client.collections.get(COLLECTIONS[coll_type]['name'])
        weight = COLLECTIONS[coll_type]['weight']

        # Get query vector using filter
        response = collection.query.fetch_objects(
            filters=Filter.by_property("nct_id").equal(query_nct_id),
            include_vector=True,
            limit=1
        )

        if not response.objects:
            continue

        query_vector = response.objects[0].vector['default']

        # Find similar trials
        similar = collection.query.near_vector(
            near_vector=query_vector,
            limit=200,  # Higher limit to find all Stage 1 trials
            return_metadata=MetadataQuery(distance=True)
        )

        # Process results
        for obj in similar.objects:
            nct_id = obj.properties['nct_id']
            if nct_id in stage2_results:
                similarity = 1 - obj.metadata.distance
                stage2_results[nct_id]['stage2_score'] += similarity * weight
                stage2_results[nct_id]['similarities'][coll_type] = similarity

    # Stage 3: Entity similarity
    st.write("### Stage 3: Computing entity similarity")
    reranker = EntityReranker()

    # Get query trial using filter on 'TrialTitle' for entities
    collection = client.collections.get('TrialTitle')
    response = collection.query.fetch_objects(
        filters=Filter.by_property("nct_id").equal(query_nct_id),
        limit=1
    )

    if response.objects:
        query_trial = response.objects[0].properties
        query_entities = reranker.extract_trial_entities(query_trial)

        # Process each trial
        st.write("Processing entity similarities...")
        for nct_id in tqdm(stage2_results.keys()):
            trial_properties = stage2_results[nct_id]['properties']
            trial_entities = reranker.extract_trial_entities(trial_properties)
            entity_sim = reranker.compute_entity_similarity(query_entities, trial_entities)
            stage2_results[nct_id]['entity_score'] = entity_sim
    else:
        # If no objects found for query
        for nct_id in stage2_results.keys():
            stage2_results[nct_id]['entity_score'] = 0

    # Stage 4: Categorical similarity
    st.write("### Stage 4: Computing categorical similarity")

    if query_nct_id in cat_index:
        query_idx = cat_index[query_nct_id]
        query_cat = cat_features[query_idx]

        # Calculate similarities for each trial
        st.write("Computing categorical similarities...")
        for nct_id in stage2_results:
            if nct_id in cat_index:
                # Get trial's categorical features
                trial_idx = cat_index[nct_id]
                trial_cat = cat_features[trial_idx]

                # Calculate cosine similarity
                num = query_cat.multiply(trial_cat).sum()
                den = np.sqrt(query_cat.multiply(query_cat).sum()) * np.sqrt(trial_cat.multiply(trial_cat).sum())

                if den > 0:
                    cat_sim = float(num / den)
                    stage2_results[nct_id]['cat_score'] = cat_sim
                else:
                    stage2_results[nct_id]['cat_score'] = 0
            else:
                stage2_results[nct_id]['cat_score'] = 0
    else:
        st.warning(f"Warning: Query trial {query_nct_id} not found in categorical index.")
        for nct_id in stage2_results:
            stage2_results[nct_id]['cat_score'] = 0

    # Final scoring
    st.write("### Computing final scores")
    final_results = []

    for nct_id, data in stage2_results.items():
        final_score = (
            0.40 * data['stage1_score'] +
            0.30 * data.get('stage2_score', 0) +
            0.20 * data.get('entity_score', 0) +
            0.10 * data.get('cat_score', 0)
        )

        final_results.append({
            'nct_id': nct_id,
            'title': data['properties'].get('study_title', 'N/A'),
            'final_score': final_score,
            'outcome_score': data['stage1_score'],
            'criteria_score': data.get('stage2_score', 0),
            'entity_score': data.get('entity_score', 0),
            'cat_score': data.get('cat_score', 0),
            'similarities': data['similarities']
        })

    # Sort and return top N
    final_results.sort(key=lambda x: x['final_score'], reverse=True)
    return final_results[:n_final]


# ----------------------------
# Main Streamlit App
# ----------------------------
def main():
    st.title("Clinical Trial Similarity Search")

    # Initialize session state
    if "condition" not in st.session_state:
        st.session_state.condition = ""
    if "intervention" not in st.session_state:
        st.session_state.intervention = ""
    if "trials_df" not in st.session_state:
        st.session_state.trials_df = pd.DataFrame()
    if "selected_nct" not in st.session_state:
        st.session_state.selected_nct = None

    # Step 1: Connect to Weaviate once
    try:
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=st.secrets["api_url"],
            auth_credentials=Auth.api_key(st.secrets["api_auth"]),
        )
    except Exception as e:
        st.error(f"Error connecting to Weaviate: {e}")
        return

    # Step 2: Gather input from user
    st.subheader("Search for Clinical Trials")
    st.session_state.condition = st.text_input(
        "Enter Condition (e.g., Ulcerative Colitis):",
        value=st.session_state.condition
    )
    st.session_state.intervention = st.text_input(
        "Enter Intervention (e.g., Adalimumab):",
        value=st.session_state.intervention
    )

    # Button to trigger search
    if st.button("Search ClinicalTrials.gov"):
        if not st.session_state.condition or not st.session_state.intervention:
            st.warning("Both Condition and Intervention are required.")
        else:
            with st.spinner("Querying ClinicalTrials.gov API..."):
                st.session_state.trials_df = query_clinical_trials(
                    st.session_state.condition,
                    st.session_state.intervention,
                    max_results=10
                )
            if st.session_state.trials_df.empty:
                st.warning("No matching trials found.")
            else:
                st.success("Trials fetched successfully!")

    # If we have trials in session state, display them
    if not st.session_state.trials_df.empty:
        display_trials(st.session_state.trials_df)

        # Let user select an NCT ID
        st.subheader("Select an NCT ID for similarity search")
        nct_options = ["Select an NCT ID"] + st.session_state.trials_df['nct_id'].unique().tolist()
        default_index = nct_options.index(st.session_state.selected_nct) if st.session_state.selected_nct in nct_options else 0

        # Use selectbox
        selected = st.selectbox("NCT ID", nct_options, index=default_index)

        # If user picks a new value, update session_state
        if selected != st.session_state.selected_nct:
            st.session_state.selected_nct = selected

        # If user picked a valid trial
        if (st.session_state.selected_nct and
            st.session_state.selected_nct != "Select an NCT ID"):

            st.write(f"**You selected:** {st.session_state.selected_nct}")

            # Verify the selected trial is in Weaviate DB
            collection = client.collections.get('TrialTitle')
            response = collection.query.fetch_objects(
                filters=Filter.by_property("nct_id").equal(st.session_state.selected_nct),
                limit=1
            )

            if not response.objects:
                st.warning(f"Selected trial {st.session_state.selected_nct} not found in the Weaviate database.")
                st.info("Please select another NCT ID.")
            else:
                # Perform multi-stage similarity
                st.write(f"Finding trials similar to **{st.session_state.selected_nct}**...")
                with st.spinner("Running multi-stage similarity search..."):
                    similar_trials = find_similar_trials(
                        query_nct_id=st.session_state.selected_nct,
                        client=client,
                        n_stage1=40,
                        n_final=10
                    )

                # Display results
                if similar_trials:
                    st.subheader("Top Similar Trials")
                    for i, trial in enumerate(similar_trials, 1):
                        st.write(f"**{i}. NCT ID:** {trial['nct_id']}")
                        st.write(f"**Title:** {trial['title']}")
                        st.write(f"**Final Score:** {trial['final_score']:.4f}")

                        with st.expander("View Component Scores"):
                            st.write(f"- **Outcome Score:** {trial['outcome_score']:.4f}")
                            st.write(f"- **Criteria Score:** {trial['criteria_score']:.4f}")
                            st.write(f"- **Entity Score:** {trial['entity_score']:.4f}")
                            st.write(f"- **Categorical Score:** {trial['cat_score']:.4f}")

                        with st.expander("View Similarity Details"):
                            for coll_type, sim_val in trial['similarities'].items():
                                st.write(f"- **{coll_type}**: {sim_val:.4f}")
                        st.write("---")
                else:
                    st.warning("No similar trials found or an error occurred.")


# Run the app
if __name__ == "__main__":
    main()
