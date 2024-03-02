import json
import subprocess

# Define the URL and output file for the curl command
file_url = "https://huggingface.co/datasets/shahules786/orca-chat/resolve/main/orca-chat-gpt4-8k.json?download=true"
output_file_name_curl = 'orca-chat-gpt4-8k.json'

# Run the curl command to download the file
subprocess.run(["curl", "-L", file_url, "-o", output_file_name_curl], check=True)

# Define the input and output file names for processing
input_file_name = output_file_name_curl  # Use the downloaded file as input
output_file_name = 'output.jsonl'  # The resulting JSONL file

# Read the input JSON data
with open(input_file_name, 'r') as input_file:
   data = json.load(input_file)

# Open the output file in write mode
with open(output_file_name, 'w') as output_file:
   # Iterate through each entry in the input data
   for entry in data:
       conversation = entry["conversation"]
       # For each conversation, create a new object and write it to the output file
       for message in conversation:
           # Construct the message object based on the provided example
           user_message = {
               "messages": [
                   {"content": message["input"], "role": "user"},
                   {"content": message["output"], "role": "assistant"}
               ]
           }
           # Write the message object as a JSON string followed by a newline
           output_file.write(json.dumps(user_message) + '\n')