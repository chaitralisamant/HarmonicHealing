# pip install -r requirements.txt

from flask import Flask, request, jsonify, render_template
from spotipy.oauth2 import SpotifyOAuth
import spotipy
import joblib
import numpy as np

app = Flask(__name__, template_folder="graphics")

# Load pretrained Random Forest model
model = joblib.load("stress_det.pkl")

track_genre = None #global var
track_id = None
artist = None
genres = []

# Spotify setup
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id="d8758ffc89ea40ba874a37fe762600fb",
    client_secret="553fc46a5c4948cd80e9fa68a9d873b8",
    redirect_uri="http://127.0.0.1:8888/callback",
    scope="user-modify-playback-state user-read-playback-state"
))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/play', methods=['POST'])
def play():
    global track_id #edit track id
    global genres
    global artist
    data = request.get_json()
    song_name = data.get("song")

    results = sp.search(q=song_name, type='track', limit=1)
    if not results['tracks']['items']:
        return jsonify({"error": "Song not found"}), 404

    track = results['tracks']['items'][0]
    track_uri = track['uri']
    results = sp.search(q=song_name, type='track', limit=1)
    track = results['tracks']['items'][0]
    track_id = track['id']
    artist_id = track['artists'][0]['id']
    artist = sp.artist(artist_id)
    genres = artist.get("genres", []) #list of strings

    devices = sp.devices()
    if not devices['devices']:
        return jsonify({"error": "No active Spotify device"}), 400

    device_id = devices['devices'][0]['id']
    sp.start_playback(device_id=device_id, uris=[track_uri])

    return jsonify({
        "message": f"Playing: {track['name']} by {track['artists'][0]['name']}"
    })

# # Starting song (used as seed track)
# SEED_TRACK_ID = "2d8JP84HNLKhmd6IYOoupQ"  # Replace with any Spotify track ID

@app.route('/predict', methods=['POST'])
def predict():
    
    global seed_track_id
    seed_track_id = track_id
    
    global genres
    seed_genres = genres
    
    data = request.get_json()
    bpm_input = float(data['bpm'])
    
    if not track_id:
        return jsonify({"error": "No seed song provided yet."}), 400

    # Predict stress level
    input = hr_to_features(bpm_input) # some guards for now, bc model is wack af
    if(bpm_input >= 130):
        stress = True
    else:
        prediction = model.predict(input)[0]
        stress = bool(prediction)
    
    #breakpoint()
    # Define tempo targets
    #tempo_target = 80 if stress else 120  # Chill or energetic
    
    chill_playlist_id = "29YdiDzSSIIEKs7agb7aWT"
    energetic_playlist_id = "7ItJydHJeasw7hj17gmqVG"

    # Get recommendations
    # recs = sp.recommendations(
    #     seed_tracks=[seed_track_id],
    #     limit=1,
    #     target_tempo=tempo_target
    # )
    if(stress):
        relevant_tracks = get_playlist_tracks(chill_playlist_id)
    elif not stress:
        relevant_tracks = get_playlist_tracks(energetic_playlist_id)
    # else:
    #     return jsonify({"error": "No track found"}), 500     

    genre_matched_tracks = []

    for item in relevant_tracks:
        #breakpoint()
        track = item['track']
        artist_id = track['artists'][0]['id']
        artist_info = sp.artist(artist_id)
        artist_genres = artist_info.get("genres", [])
        if any(genre in seed_genres for genre in artist_genres):
            genre_matched_tracks.append(track)
        elif(artist_info['name'] == artist['name']):
            genre_matched_tracks.append(track) # just in case no genre is recognized

    #breakpoint()
    
    if not genre_matched_tracks:
        return jsonify({
            # "error": "No genre-matched track found",
            "seed_genres": ", ".join(seed_genres)
        }), 500

    chosen_track = np.random.choice(genre_matched_tracks)
    chosen_track_uri = chosen_track['uri']
    chosen_track_name = chosen_track['name']
    chosen_artist_id = chosen_track['artists'][0]['id']
    chosen_artist_info = sp.artist(chosen_artist_id)
    chosen_artist = chosen_artist_info['name']
    
    # Play on first available device
    devices = sp.devices()
    if devices['devices']:
        device_id = devices['devices'][0]['id']
        sp.start_playback(device_id=device_id, uris=[chosen_track_uri])
    else:
        return jsonify({"error": "No active Spotify device"}), 400

    return jsonify({
        "prediction": "Stress" if stress else "No Stress",
        "song": f"{chosen_track_name} by {chosen_artist}",
        "uri": chosen_track_uri
    })

def hr_to_features(hr):
    
    # estimating the values from heart rate alone
    
    mean_rr = 60000/hr
    median_rr = 60000/hr
    
    sdrr = 0.05 * (mean_rr)
    
    sdrr_rmssd = 1
    mean_rel_rr = 1
    median_rel_rr = 1
    sdrr_rel_rr = 0.05
    
    features = [[mean_rr, median_rr, sdrr, sdrr_rmssd, hr, mean_rel_rr, median_rel_rr, sdrr_rel_rr]]
    
    scaler = joblib.load("scaler.pkl")
    pca = joblib.load("pca.pkl")

    standardized = scaler.transform(features)
    pca_features = pca.transform(standardized)
    
    return pca_features

def get_playlist_tracks(playlist_id):
    results = sp.playlist_tracks(playlist_id)
    return results['items']

if __name__ == '__main__':
    app.run(debug=True)
