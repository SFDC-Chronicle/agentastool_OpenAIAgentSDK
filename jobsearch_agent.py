import asyncio
import dotenv

from pydantic import BaseModel, Field

from agents import Agent, Runner, WebSearchTool, ModelSettings

"""
This example shows structured input for agent-as-tool calls.
"""


dotenv.load_dotenv()


class SearchParameterInput(BaseModel):
    jobtitle: str = Field(
        description=" Searching specific roles or tech stacks.")
    location: str = Field(
        description="Applicant wants to work from where like Remote, hybrid, on-site i.e. from any specific city")
    jobtype: str = Field(
        description="Applicant wants what type of job like Full-time, part-time, contract, freelance, or internship.")
    experiencelevel: int = Field(
        description="Applicant experience level in years")
    minsalary: int = Field(description="Minimum salary expected by Applicant")
    maxsalary: int = Field(description="Maximum salary expected by Applicant")


jobsearch = Agent(
    name="Parameter Setter",
    instructions=(
        "Search the job based on the search parameter provide by user, to search current job listing use WebSearchTool tool"
        "If the jobtitle, location, experiencelevel is not clear, ask the user for clarification."
    ),
    tools=[WebSearchTool()],
)

orchestrator = Agent(
    name="orchestrator",
    instructions=(
        "You are a task dispatcher. Always call the tool with sufficient input. "
        "Do not handle the job search by yourself."
    ),
    tools=[
        jobsearch.as_tool(
            tool_name="job_search",
            tool_description=(
                "Search the job based as per user preferences. Provide jobtitle, location, jobtype"
                "and experiencelevel."
            ),
            parameters=SearchParameterInput,
            include_input_schema=True,
        )
    ],
    model_settings=ModelSettings(tool_choice="job_search")
)


async def main() -> None:
    query = 'I am a 5 years experience OpenAI developer, looking for Full-time job in New York, USA'

    response2 = await Runner.run(orchestrator, query)
    print(f"Job Search agent as tool: {response2.final_output}")


if __name__ == "__main__":
    asyncio.run(main())
