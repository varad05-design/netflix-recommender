import json
import pickle
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Netflix Recommender",
    page_icon="🍿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# 2. Custom CSS for Netflix Dark Theme
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Dark Theme Background */
    .stApp {
        background-color: #141414;
        color: #FFFFFF;
    }
    
    /* Netflix Header Styling */
    .netflix-title {
        color: #E50914;
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: -1px;
        margin-bottom: 0px;
    }
    .netflix-subtitle {
        color: #A0A0A0;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    /* Recommendation Card Styling */
    .movie-card {
        background-color: #1F1F1F;
        border-radius: 8px;
        padding: 18px;
        margin-bottom: 20px;
        border: 1px solid #2B2B2B;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .movie-card:hover {
        border-color: #E50914;
        transform: translateY(-3px);
    }
    
    /* Badge tags */
    .badge-type {
        background-color: #E50914;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: bold;
        margin-right: 6px;
    }
    .badge-year {
        background-color: #333333;
        color: #CCCCCC;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        margin-right: 6px;
    }
    .badge-genre {
        background-color: #262626;
        color: #00D46A;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        border: 1px solid #00D46A;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# 3. Data Loading (Cached for Speed)
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
  df = pd.read_csv('netflix_titles.csv')
  with open('cosine_sim.pkl', 'rb') as f:
    cosine_sim = pickle.load(f)
  with open('indices.json', 'r') as f:
    indices_raw = json.load(f)
  indices = pd.Series(indices_raw)
  return df, cosine_sim, indices


try:
  df, cosine_sim, indices = load_data()
except Exception as e:
  st.error(f'Failed to load datasets: {e}')
  st.stop()


# -----------------------------------------------------------------------------
# 4. Recommendation Logic with Filtering
# -----------------------------------------------------------------------------
def get_recommendations(
    title, num_recommendations=6, type_filter='All', genre_filter='All'
):
  idx = indices[title]
  if isinstance(idx, pd.Series):
    idx = idx.iloc[0]

  sim_scores = list(enumerate(cosine_sim[idx]))
  sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

  # Get top candidate pool
  candidate_indices = [i[0] for i in sim_scores[1:100]]
  candidates = df.iloc[candidate_indices].copy()

  # Apply Sidebar Filters
  if type_filter != 'All':
    candidates = candidates[candidates['type'] == type_filter]

  if genre_filter != 'All' and 'primary_genre' in candidates.columns:
    candidates = candidates[candidates['primary_genre'] == genre_filter]

  return candidates.head(num_recommendations)


# -----------------------------------------------------------------------------
# 5. Sidebar Controls & Preferences
# -----------------------------------------------------------------------------
st.sidebar.image(
    'https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg',
    width=160,
)
st.sidebar.markdown('---')
st.sidebar.subheader('⚙️ Filter & Customization')

num_recs = st.sidebar.slider(
    'Number of Recommendations:', min_value=3, max_value=12, value=6, step=3
)

type_filter = st.sidebar.radio(
    'Content Type:', ['All', 'Movie', 'TV Show'], horizontal=True
)

available_genres = ['All']
if 'primary_genre' in df.columns:
  available_genres += sorted(df['primary_genre'].dropna().unique().tolist())

genre_filter = st.sidebar.selectbox('Genre Filter:', available_genres)

st.sidebar.markdown('---')
st.sidebar.caption('Powered by Cosine Similarity & Natural Language Processing')

# -----------------------------------------------------------------------------
# 6. Main Header & Selection Section
# -----------------------------------------------------------------------------
st.markdown(
    '<p class="netflix-title">NETFLIX RECOMMENDER</p>', unsafe_allow_html=True
)
st.markdown(
    '<p class="netflix-subtitle">Discover movies & TV shows based on your favorite titles</p>',
    unsafe_allow_html=True,
)

col1, col2 = st.columns([3, 1])

with col1:
  selected_title = st.selectbox(
      '🔍 Choose or search for a title you like:',
      options=sorted(df['title'].unique()),
      index=0,
  )

with col2:
  st.write('##')
  recommend_button = st.button(
      '🚀 Get Recommendations', use_container_width=True, type='primary'
  )

# Display stats for the selected title
selected_item = df[df['title'] == selected_title].iloc[0]

with st.expander(
    f"ℹ️ **Selected Title Details:** {selected_item['title']}", expanded=False
):
  m_col1, m_col2, m_col3 = st.columns(3)
  m_col1.metric("Type", selected_item.get("type", "N/A"))
  m_col2.metric("Release Year", str(selected_item.get("release_year", "N/A")))
  if "primary_genre" in df.columns:
    m_col3.metric("Genre", selected_item.get("primary_genre", "N/A"))

  st.write(
      f"**Description:** {selected_item.get('description', 'No overview available.')}"
  )

st.markdown('---')

# -----------------------------------------------------------------------------
# 7. Recommendations Output (Grid Display)
# -----------------------------------------------------------------------------
if recommend_button:
  with st.spinner('Curating recommendations...'):
    results = get_recommendations(
        selected_title,
        num_recommendations=num_recs,
        type_filter=type_filter,
        genre_filter=genre_filter,
    )

  if results.empty:
    st.warning(
        'No matches found matching your sidebar filter criteria. Try adjusting the type or genre filters!'
    )
  else:
    st.subheader(f'🎬 Top {len(results)} Titles Similar to "{selected_title}":')

    # Display in a 3-column grid layout
    cols_per_row = 3
    results_list = [
        results.iloc[i : i + cols_per_row]
        for i in range(0, len(results), cols_per_row)
    ]

    for row in results_list:
      grid_cols = st.columns(cols_per_row)
      for idx, (_, item) in enumerate(row.iterrows()):
        with grid_cols[idx]:
          genre_text = item.get('primary_genre', 'N/A')
          type_text = item.get('type', 'N/A')
          year_text = item.get('release_year', 'N/A')

          st.markdown(
              f"""
                        <div class="movie-card">
                            <h3>{item['title']}</h3>
                            <p style="margin-top:-10px;">
                                <span class="badge-type">{type_text}</span>
                                <span class="badge-year">{year_text}</span>
                                <span class="badge-genre">{genre_text}</span>
                            </p>
                        </div>
                        """,
              unsafe_allow_html=True,
          )

          with st.expander('Read Synopsis'):
            st.write(
                item.get('description', 'No synopsis available for this title.')
            )
            if 'cast' in item and pd.notna(item['cast']):
              st.caption(f"**Cast:** {item['cast']}")