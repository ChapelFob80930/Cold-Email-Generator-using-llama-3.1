import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv
from serpapi.google_search import GoogleSearch

load_dotenv()

class Chain:
    def __init__(self):
        self.llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, groq_api_key=os.getenv("API_KEY"))

    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE OR JOB DESCRIPTION:
            {page_data}
            ### INSTRUCTION:
            The text provided is either from a career page or a job description.
            Your job is to extract a single job posting and return it in JSON format containing the 
            following keys: role, experience, skills, and description.
            skills value is not a dictionary, rather just contains the skills and the skills should be single values eg."skills":"kubernetes","Docker","Python" and not one complete sentence like "skills":  "Proficient in running applications on Cloud (AWS, Azure, or equivalent) using Kubernetes and Docker".
            Only return a valid JSON object without wrapping it in a list and no examples required.
            ### VALID JSON (NO PREAMBLE):
            """
        )
        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={"page_data": cleaned_text})
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse jobs.")
        return res if isinstance(res, list) else [res]

    def write_mail(self, job, links):
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}

            ### INSTRUCTION:
            You are Sarbojit, a business development executive at AtliQ. AtliQ is an AI & Software Consulting company dedicated to facilitating
            the seamless integration of business processes through automated tools. 
            Over our experience, we have empowered numerous enterprises with tailored solutions, fostering scalability, 
            process optimization, cost reduction, and heightened overall efficiency. 
            Your job is to write a cold email to the client regarding the job mentioned above describing the capability of AtliQ 
            in fulfilling their needs.
            Also add the most relevant ones from the following links to showcase Atliq's portfolio: {link_list}
            Remember you are Sarbojit, BDE at AtliQ. 
            Do not provide a preamble.
            ### EMAIL (NO PREAMBLE):
            """
        )
        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description": str(job), "link_list": links})
        return res.content

    def search_jobs(self, job_title, location):
        params = {
            "engine": "google_jobs",
            "q": job_title,
            "location": location,
            "hl": "en",
            "api_key": os.getenv("SERPAPI_API_KEY")
        }

        search = GoogleSearch(params)
        results = search.get_dict()

        jobs = results.get("jobs_results", [])

        # Extract relevant fields and extract skills using LLM
        extracted_jobs = []
        for job in jobs:
            job_description = job.get("description", "")
            
            # Send description to extract_jobs method to get structured data
            extracted_job = self.extract_jobs(job_description)

            if extracted_job:
                # Add extracted skills and other job info to the final dictionary
                job_details = {
                    "title": job.get("title", ""),
                    "company_name": job.get("company_name", ""),
                    "location": job.get("location", ""),
                    "description": job_description,
                    "skills": extracted_job[0].get("skills", []),  # Extracted skills
                    "share_link": job.get("share_link", "")
                }
                extracted_jobs.append(job_details)

        return extracted_jobs
