import streamlit as st
from transformers import pipeline

# Cache the model loading to avoid reloading on every interaction
@st.cache_resource
def load_generator():
    return pipeline('text-generation', model='bigscience/bloom-560m')

generator = load_generator()

st.title("🌊 Ocean Poem Generator")

# Input area for the prompt with a default suggestion
prompt = st.text_area("Enter your prompt:", "Write a small poem about the ocean", height=100)

if st.button("Generate Poem"):
    with st.spinner("Generating..."):
        results = generator(prompt, max_length=60, num_return_sequences=1)
        poem = results[0]['generated_text']
        st.markdown("### Here's your generated poem:")
        st.markdown(f"> {poem}")
