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

    if submit_button:
        if not job_title or not location:
            st.error("Please enter both job title and location.")
            return

        try:
            # Search jobs based on user input
            st.info(f"🔎 Searching for related jobs in {location}.... ")
            jobs = llm.search_jobs(job_title, location)

            if not jobs:
                st.warning("⚠️ No jobs found. Try a different search.")
                return

            # Display the list of jobs for user to choose from
            st.write("✅ Found the following jobs:")
            job_choices = {}
            for idx, job in enumerate(jobs):
                job_title_display = f"{job['title']} at {job['company_name']} ({job['location']})"
                job_choices[idx + 1] = job  # Storing job data with index as key
                st.write(f"{idx + 1}. {job_title_display}")

            job_selection = st.selectbox("🎯 Select a job to generate cold email", options=list(job_choices.keys()))
            selected_job = job_choices[job_selection]

            # Load portfolio and get relevant links based on job skills
            skills = selected_job.get('skills')
            links = portfolio.query_links(skills)
            st.write(links)
            # Generate cold email for selected job
            st.info("✉️ Generating cold email...")
            email = llm.write_mail(selected_job, links)
            st.subheader("📄 Generated Cold Email:")
            st.code(email, language='markdown')

        except Exception as e:
            st.error(f"❌ An error occurred: {e}")


# Run the Streamlit app
if __name__ == "__main__":
    create_streamlit_app(llm, portfolio)
