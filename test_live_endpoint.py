"""
Live /analyze endpoint test.
Requires app.py to already be running on port 8000.
Run with: .\\cenv\\Scripts\\python.exe test_live_endpoint.py
"""
import requests, os, sys

sys.path.insert(0, '.')

# --- Find test image ---
img_path = None
for root, dirs, files in os.walk('.'):
    if '.git' in root:
        continue
    for fname in files:
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            fpath = os.path.join(root, fname)
            img_path = fpath
            if 'cancer' in fname.lower():
                break
    if img_path and 'cancer' in img_path.lower():
        break

if img_path is None:
    print("ERROR: No image found for testing.")
    sys.exit(1)

print(f"Using image: {img_path}")

# --- POST to /analyze ---
with open(img_path, 'rb') as f:
    img_bytes = f.read()

files = {'file': (os.path.basename(img_path), img_bytes, 'image/jpeg')}
print("POSTing to http://127.0.0.1:8000/analyze ...")
resp = requests.post('http://127.0.0.1:8000/analyze', files=files, timeout=60)

print(f"HTTP Status: {resp.status_code}")

if resp.status_code != 200:
    print("ERROR:", resp.text[:500])
    sys.exit(1)

data = resp.json()
print(f"label       : {data.get('label')}")
print(f"percent     : {data.get('percent', 0):.2f}%")
print(f"class_id    : {data.get('class_id')}")

gradcam_val = data.get('gradcam_image')
if gradcam_val and len(gradcam_val) > 100:
    print(f"gradcam_image: PRESENT — {len(gradcam_val)} chars base64")
    print()
    print("=" * 55)
    print("RESULT: GRAD-CAM IS WORKING CORRECTLY")
    print("=" * 55)
else:
    print(f"gradcam_image: {repr(gradcam_val)}")
    print()
    print("RESULT: GRAD-CAM FAILED — gradcam_image is empty/None")
    sys.exit(1)
