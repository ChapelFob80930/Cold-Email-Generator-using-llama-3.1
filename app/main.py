import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from chains import Chain
from portfolio import Portfolio
from utils import clean_text

# Initialize Chain and Portfolio instances
llm = Chain()
portfolio = Portfolio()
portfolio.load_portfolio()

def create_streamlit_app(llm, portfolio):
    st.title("📧 Cold Mail Generator")

    # User input for job search
    job_title = st.text_input("Enter job title (e.g., Software Engineer):")
    location = st.text_input("Enter job location (e.g., Paris, France):")
    submit_button = st.button("Search Jobs")

    # Job Search Logic (Only runs once, doesn't reset)
    if submit_button:
        if not job_title or not location:
            st.error("Please enter both job title and location.")
            return

        try:
            st.info(f"🔎 Searching for related jobs in {location}.... ")

            # Check if jobs are already in session state, if not, search and store them
            if "jobs" not in st.session_state:
                jobs = llm.search_jobs(job_title, location)
                if not jobs:
                    st.warning("⚠️ No jobs found. Try a different search.")
                    return
                st.session_state.jobs = jobs
            else:
                jobs = st.session_state.jobs  # Use the stored jobs

            # Display the list of jobs for user to choose from
            st.write("✅ Found the following jobs:")
            job_choices = {}
            for idx, job in enumerate(jobs):
                job_title_display = f"{job['title']} at {job['company_name']} ({job['location']})"
                job_choices[idx + 1] = job  # Store job data with index as key
                st.write(f"{idx + 1}. {job_title_display}")

            # Select job from dropdown
            job_selection = st.selectbox("🎯 Select a job to generate cold email", options=list(job_choices.keys()))

            # If there's a new selection, update session state
            selected_job = job_choices[job_selection]
            st.session_state.selected_job = selected_job

            # Load portfolio and get relevant links based on job skills (only update on new selection)
            skills = selected_job.get('skills')
            
            if "links" not in st.session_state or st.session_state.selected_job != selected_job:
                links = portfolio.query_links(skills)
                st.session_state.links = links

            # Generate cold email for selected job
            st.info("✉️ Generating cold email...")
            email = llm.write_mail(selected_job, st.session_state.links)
            st.subheader("📄 Generated Cold Email:")
            st.code(email, language='markdown')

        except Exception as e:
            st.error(f"❌ An error occurred: {e}")


# Run the Streamlit app
if __name__ == "__main__":
    create_streamlit_app(llm, portfolio)
