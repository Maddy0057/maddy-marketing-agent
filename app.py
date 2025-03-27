from flask import Flask, render_template, request, jsonify
import warnings
warnings.filterwarnings('ignore')
from crewai import Agent, Task, Crew, LLM
import google.generativeai as genai
import os
import markdown

app = Flask(__name__)

# Load API key (replace with your actual key or use environment variable)
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)
model_name = 'gemini-2.0-flash'
gemini_llm = LLM(model=f"gemini/{model_name}", api_key=api_key)

@app.route('/', methods=['GET', 'POST'])
def index():
    results = None
    if request.method == 'POST':
        try:
            # Extract form data
            prompt = request.form['prompt']
            cost_input = int(request.form['usercost'])
            currency = request.form['currency']
            input_place = request.form['place']

            # Input validation
            if not prompt.strip() or cost_input <= 0 or not input_place.strip():
                raise ValueError("Invalid input: Prompt, budget, and location must be valid.")

            # Convert cost_input to INR if USD
            if currency == "USD":
                exchange_rate = 87  # Hardcoded for now (1 USD = ₹87 as of March 2025)
                cost_input = cost_input * exchange_rate

            # Define Agents
            user_interface_agent = Agent(
                role="User Interface Agent",
                goal="Interpret queries",
                backstory="Extracts industry, goals, location, and budget from user input.",
                llm=gemini_llm,
                verbose=True
            )
            data_management_agent = Agent(
                role="Data Management Agent",
                goal="Collect local data",
                backstory="Gathers location-specific trends.",
                llm=gemini_llm,
                verbose=True
            )
            market_analysis_agent = Agent(
                role="Market Analysis Agent",
                goal="Analyze local trends",
                backstory="Focuses on local competitors.",
                llm=gemini_llm,
                verbose=True
            )
            customer_insights_agent = Agent(
                role="Customer Insights Agent",
                goal="Understand local customers",
                backstory="Profiles local buyers.",
                llm=gemini_llm,
                verbose=True
            )
            predictive_modeling_agent = Agent(
                role="Predictive Modeling Agent",
                goal="Forecast local trends",
                backstory="Predicts market trends.",
                llm=gemini_llm,
                verbose=True
            )
            creative_strategy_agent = Agent(
                role="Creative Strategy Agent",
                goal="Generate practical ideas",
                backstory="Crafts location-specific tactics.",
                llm=gemini_llm,
                verbose=True
            )
            performance_analysis_agent = Agent(
                role="Performance Analysis Agent",
                goal="Evaluate feasibility",
                backstory="Ensures local practicality.",
                llm=gemini_llm,
                verbose=True
            )
            local_adapter_agent = Agent(
                role="Local Market Adapter Agent",
                goal="Localize strategies",
                backstory="Tailors ideas to the specified location.",
                llm=gemini_llm,
                verbose=True
            )
            cost_constraint_agent = Agent(
                role="Cost Constraint Agent",
                goal="Fit strategies within budget",
                backstory="Ensures ideas stay within cost_input.",
                llm=gemini_llm,
                verbose=True
            )
            risk_management_agent = Agent(
                role="Risk Management Agent",
                goal="Assess risks and success rate",
                backstory="Evaluates risks based on budget and location.",
                llm=gemini_llm,
                verbose=True
            )

            # Define Tasks
            handle_user_query = Task(
                description=f"Interpret the prompt: '{prompt}' Extract the industry, goal, location, and budget.",
                expected_output=f"Industry: Extracted from prompt, Goal: Extracted from prompt, Location: {input_place}, Budget: ₹{cost_input}",
                agent=user_interface_agent
            )
            collect_data = Task(
                description=f"Simulate collecting data for the business specified in '{prompt}' in {input_place}. Include local trends, customer data, and location-specific strategies.",
                expected_output=f"Local trends, customer profiles, and {input_place}-specific data for the specified business.",
                agent=data_management_agent,
                context=[handle_user_query]
            )
            analyze_market = Task(
                description=f"Analyze market trends and competitors for the business specified in '{prompt}' in {input_place}.",
                expected_output=f"{input_place} market trends and competitor strategies for the specified business.",
                agent=market_analysis_agent,
                context=[collect_data]
            )
            analyze_customers = Task(
                description=f"Profile customers for the business specified in '{prompt}' in {input_place}.",
                expected_output=f"{input_place} customer segments and preferences for the specified business.",
                agent=customer_insights_agent,
                context=[collect_data]
            )
            predict_trends = Task(
                description=f"Predict trends for the business specified in '{prompt}' in {input_place} over 6-12 months.",
                expected_output=f"Future trends and opportunities in {input_place} for the specified business.",
                agent=predictive_modeling_agent,
                context=[analyze_market, analyze_customers]
            )
            generate_ideas = Task(
                description=f"Generate 3 practical marketing ideas for the business specified in '{prompt}' in {input_place}, using location-specific strategies.",
                expected_output=f"3 {input_place}-specific, actionable ideas for the specified business.",
                agent=creative_strategy_agent,
                context=[analyze_market, analyze_customers, predict_trends]
            )
            review_performance = Task(
                description=f"Evaluate the 3 ideas for feasibility and impact in {input_place}’s market for the business specified in '{prompt}'.",
                expected_output="Feedback on feasibility and impact for each idea.",
                agent=performance_analysis_agent,
                context=[generate_ideas]
            )
            localize_strategies = Task(
                description=f"Adapt the 3 ideas to {input_place}’s street-level and market realities for the business specified in '{prompt}', emphasizing practical tactics.",
                expected_output=f"Localized versions of the 3 ideas for {input_place}.",
                agent=local_adapter_agent,
                context=[generate_ideas, review_performance]
            )
            constrain_costs = Task(
                description=f"Adapt the 3 localized ideas to fit within a budget of ₹{cost_input} for {input_place} for the business specified in '{prompt}'. Estimate costs for each idea and ensure total stays under budget.",
                expected_output=f"3 cost-constrained ideas for {input_place} with estimated costs totaling ≤ ₹{cost_input}.",
                agent=cost_constraint_agent,
                context=[localize_strategies]
            )
            assess_risks = Task(
                description=f"Assess risks for the 3 cost-constrained ideas in {input_place} with a budget of ₹{cost_input} for the business specified in '{prompt}'. Evaluate market, operational, and financial risks based on local conditions and budget. Provide a success rate (0-100%) for each idea.",
                expected_output=f"Risk assessment and success rate for each of the 3 ideas in {input_place}.",
                agent=risk_management_agent,
                context=[constrain_costs]
            )

            # Define Crew
            crew = Crew(
                agents=[user_interface_agent, data_management_agent, market_analysis_agent, customer_insights_agent,
                        predictive_modeling_agent, creative_strategy_agent, performance_analysis_agent,
                        local_adapter_agent, cost_constraint_agent, risk_management_agent],
                tasks=[handle_user_query, collect_data, analyze_market, analyze_customers, predict_trends,
                       generate_ideas, review_performance, localize_strategies, constrain_costs, assess_risks],
                verbose=True
            )

            # Run Crew
            crew_result = crew.kickoff(inputs={"query": prompt})
            full_package = ""
            for task in crew.tasks:
                task_output = task.output.raw if task.output else "Task not completed"
                full_package += f"### {task.agent.role}\n{task_output}\n\n"
            results = full_package

            # Standalone Gemini Call
            try:
                model = genai.GenerativeModel(model_name)
                prompt_gemini = f"{prompt} in {input_place} with a budget of {cost_input} INR?"
                response = model.generate_content(prompt_gemini)
                results += f"\n### Simple Oral Context\n{response.text}"
            except Exception as e:
                results += f"\n### Gemini Error\nFailed to generate response: {str(e)}"

            # Convert Markdown to HTML
            results = markdown.markdown(results)

        except Exception as e:
            results = markdown.markdown(f"### Error\nAn error occurred: {str(e)}")

    # Render the external index.html file
    return render_template('index.html', results=results)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
