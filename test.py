import time
import ollama

IMAGE = r"C:\Users\Admin\OneDrive\Pictures\PYTHON\Screenshot 2026-10-15 153045.png"   # change to any file in cache\images

start = time.time()
response = ollama.chat(
    model="moondream:1.8b",
    messages=[{
        "role": "user",
        "content": "Describe this image in detail. If it is a chart, say what type of chart it is and describe the trend.",
        "images": [IMAGE],
    }],
    options={"temperature": 0, "num_predict": 150, "num_ctx": 2048},
)
print(f"Took {time.time() - start:.0f} seconds\n")
print(response["message"]["content"])