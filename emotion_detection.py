
import requests
import json

def emotion_detector(text_to_analyze):
    # Define the URL for the emotion detection API
    url = "https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"

    # Create the payload with the text to be analyzed
    myobj = {"raw_document": {"text": text_to_analyze}}

    # Set the headers with the required model ID for the API
    headers = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}

    try:
        # Make a POST request to the API with the payload and headers
        response = requests.post(url, json=myobj, headers=headers)
        response.raise_for_status()  # Raise an error for HTTP issues

        # Parse the response from the API
        response_data = json.loads(response.text)
        emotions = response_data.get("emotionPredictions", [{}])[0].get("emotion", {})

        # emotions
        emotions = {
            "anger": emotions.get("anger", 0),
            "disgust": emotions.get("disgust", 0),
            "fear": emotions.get("fear", 0),
            "joy": emotions.get("joy", 0),
            "sadness": emotions.get("sadness", 0),
        }

        # dominant emotion
        dominant_emotion = max(emotions, key=emotions.get)
        emotions["dominant_emotion"] = dominant_emotion

        return emotions
    except requests.exceptions.RequestException as error:
        return {"error": str(error)}