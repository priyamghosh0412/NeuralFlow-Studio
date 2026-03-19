from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

class LangChainRunner:
    def __init__(self):
        self.node_map = {}

    def run_flow(self, flow_data):
        """
        Parses the flow data (nodes and edges) and executes the corresponding LangChain workflow.
        
        Args:
            flow_data (dict): Contains 'nodes' and 'edges'.
            
        Returns:
            str: The output of the flow execution.
        """
        nodes = flow_data.get('nodes', [])
        edges = flow_data.get('edges', [])
        inputs = flow_data.get('inputs', {})

        # 1. Identify Components
        # For this prototype, we assume a linear chain: Input -> Prompt -> Model -> Output
        # We need to find the specific configuration for each.
        
        prompt_template = None
        model_name = "llama3" # Default
        temperature = 0.7
        
        for node in nodes:
            node_type = node.get('type') # You'll need to send 'type' from frontend
            data = node.get('data', {})
            
            if node_type == 'prompt-template':
                # Extract template string
                # Expecting 'template' in data
                template_str = data.get('template', "You are a helpful assistant.\nUser: {user_input}")
                prompt_template = ChatPromptTemplate.from_template(template_str)
            
            elif node_type == 'local-llm':
                model_name = data.get('model', 'llama3')
                temperature = float(data.get('temperature', 0.7))

        # 2. Construct Chain
        if not prompt_template:
            return "Error: No Prompt Template node found or connected."

        llm = ChatOllama(model=model_name, temperature=temperature)
        output_parser = StrOutputParser()

        # Simple LCEL Chain
        chain = prompt_template | llm | output_parser

        # 3. Execute
        # We assume the input key is 'user_input' based on the default template
        try:
            result = chain.invoke(inputs)
            return result
        except Exception as e:
            return f"Error executing flow: {str(e)}"
