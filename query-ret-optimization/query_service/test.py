from groq import Groq
import os
client = Groq(api_key=os.environ["GROQ_API_KEY"])
for m in client.models.list().data:
    print(m.id)