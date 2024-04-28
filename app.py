from flask import Flask, render_template, request
import requests
from PIL import Image
import io
import os
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    prompt = request.form['prompt']
    image_filenames = generate_images(prompt)
    return render_template('results.html', image_filenames=image_filenames)

def generate_image_dalle3(prompt):
    openai_client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
    data = openai_client.images.generate(model="dall-e-3", prompt=prompt, n=1, size="1024x1024")
    image_url = data.data[0].url
    response = requests.get(image_url)
    if response.status_code == 200:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")  # Generate timestamp
        filename = f"dalle_{timestamp}.jpg"  
        with open('static/'+filename, 'wb') as f:
            f.write(response.content)
        print(f"Image saved as {filename}")
    else:
        print("Failed to download image")
    return filename

def generate_image_sd(prompt):
    try:
        r = requests.post('https://clipdrop-api.co/text-to-image/v1', files={'prompt': (None, prompt, 'text/plain')},
                          headers={'x-api-key':os.environ.get('SD_API_KEY') })
        if r.ok:
            image = Image.open(io.BytesIO(r.content))
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")  # Generate timestamp
            filename = f"SD_{timestamp}.jpg"
            image.save('static/'+filename)
            print(f"SD Image saved as {filename}")
        else:
            print(f"Failed to generate image: {r.status_code}")
    except Exception as e:
        print(f"An error occurred: {e}")
    return filename

def generate_images(prompt):
    model_functions = [
        ("DALLE3", generate_image_dalle3),
        ("Stable Diffusion", generate_image_sd),
    ]
    image_filenames = []
    for model, func in model_functions:
        filename = func(prompt)
        image_filenames.append((model, filename))  
    return image_filenames


if __name__ == '__main__':
    app.run(debug=True)
