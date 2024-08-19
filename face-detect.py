from flask import Flask, request, jsonify
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import face_recognition
import json
from flask_cors import CORS
import logging

app = Flask(__name__)

# Allow requests only from localhost:3000
CORS(app)





def decode_image(image_data):
    """Decode the base64 image data to a numpy array."""
    header, encoded = image_data.split(",", 1)
    decoded = base64.b64decode(encoded)
    image = Image.open(BytesIO(decoded))
    image = image.convert('RGB')  # Ensure image is in RGB format
    return np.array(image)

def convert_json_to_encoding(json_data):
    """Convert JSON data to NumPy array or other encoding format."""
    return np.array(json_data)

def convert_encoding_to_json(encoding):
    """Convert encoding to JSON serializable format."""
    if isinstance(encoding, np.ndarray):
        return encoding.tolist()  # Convert NumPy array to list
    elif isinstance(encoding, list):
        return encoding
    return encoding



@app.route('/register', methods=['POST'])
def register_face():
    """Register a face by storing its encoding with a unique ID."""
    data = request.get_json()
    image_data = data.get('image')
    encodedFace= data.get('allRecognizeFace')    
    if not image_data:
        return jsonify({"error": "No image data provided"}), 400

    # Decode the image
    image = decode_image(image_data)

    # Process the image for face recognition
    try:
        face_locations = face_recognition.face_locations(image)
        if len(face_locations) > 0:
            # Extract face encodings from the image
            face_encodings = face_recognition.face_encodings(image, face_locations)


            stored_encodings = encodedFace
            for encoding in face_encodings:
                  # Check if the face encoding is already stored
                for stored_encoding in stored_encodings:
                    stored_encoding_data = convert_json_to_encoding(json.loads(stored_encoding['Encoded']))
                    matches = face_recognition.compare_faces([stored_encoding_data], encoding)
                    if True in matches:
                        # Face is already registered
                        return jsonify({
                            "success":"false",
                            "message": f"Identiy face {stored_encoding['UserID']} already registered",
                            "identity":stored_encoding['UserID'],
                        }), 409  # Conflict status code
                    

            # Save the face encoding
            for encoding in face_encodings:
                # save_face_encoding(encoding);

             return jsonify({
                "success":"true",
                "message": "Face registered successfully",
                "recognize":convert_encoding_to_json(encoding)

            }), 200
        else:
            return jsonify({"message": "No face detected"}), 400
    except Exception as e:
        return jsonify({"message": str(e)}), 500

@app.route('/verify', methods=['POST'])
def verify_face():
    """Verify a face by comparing its encoding with stored encodings."""
    data = request.get_json()
    image_data = data.get('image')
    encodedFace= data.get('allRecognizeFace')    

    

    if not image_data:
        return jsonify({"error": "No image data provided"}), 400

    # Decode the image
    try:
        image = decode_image(image_data)
    except Exception as e:
        return jsonify({"error": f"Image decoding failed: {str(e)}"}), 400

    # Process the image for face recognition
    try:
        face_locations = face_recognition.face_locations(image)
        if len(face_locations) > 0:
            # Extract face encodings from the captured image
            captured_face_encodings = face_recognition.face_encodings(image, face_locations)
            if len(captured_face_encodings) == 0:
                return jsonify({"error": "No face encoding found in captured image"}), 400

            # Verify against stored encodings
            stored_encodings = encodedFace
            for encoding in captured_face_encodings:
                for stored_entry in stored_encodings:
                    # stored_encoding = stored_entry['Encoded']
                    stored_encoding =  convert_json_to_encoding(json.loads(stored_entry['Encoded']))

                    app.logger.info('encoded: %s', stored_encoding)

                    matches = face_recognition.compare_faces([stored_encoding], encoding)
                    if True in matches:
                        return jsonify({
                            "success": "Face verified successfully",
                             "identity": stored_entry["UserID"],
                            "record": [convert_encoding_to_json(stored_entry['Encoded']) for stored_entry in stored_encodings]
                        }), 200

            return jsonify({"error": "Face not recognized"}), 401

        else:
            return jsonify({"error": "No face detected"}), 400
    except Exception as e:
        return jsonify({"error": f"Face recognition failed: {str(e)}"}), 500



if __name__ == '__main__':
    app.run(debug=True)
