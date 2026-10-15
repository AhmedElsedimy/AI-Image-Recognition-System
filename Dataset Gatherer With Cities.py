import os
import requests
import time
import random
import csv
import hashlib
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

# === COUNTRY URBAN CENTERS (WITH CITY NAMES) ===
urban_centers = {
    "TR": [
        (41.0082, 28.9784, "Istanbul"), (39.9208, 32.8541, "Ankara"),
        (38.4237, 27.1428, "Izmir"), (37.8714, 32.4846, "Konya"),
        (36.8969, 30.7133, "Antalya"), (40.1950, 29.0602, "Bursa"),
        (37.0000, 35.3213, "Adana"), (41.2867, 36.33, "Samsun"),
        (37.7648, 30.5566, "Isparta")
    ],
    "JP": [
        (35.6895, 139.6917, "Tokyo"), (34.6937, 135.5022, "Osaka"),
        (35.0116, 135.7681, "Kyoto")
    ],
    "US": [
        (40.7128, -74.0060, "New York"), (34.0522, -118.2437, "Los Angeles"),
        (41.8781, -87.6298, "Chicago")
    ],
    "AE": [
        (24.4539, 54.3773, "Abu Dhabi"), (25.2048, 55.2708, "Dubai"),
        (25.3463, 55.4209, "Sharjah"), (25.3782, 55.5007, "Ajman"),
        (25.6866, 55.7766, "Umm Al Quwain"), (25.8007, 55.9762, "Ras Al Khaimah"),
        (25.1524, 56.3345, "Fujairah"), (25.2885, 55.8815, "Al Dhaid"),
        (25.276987, 55.296249, "Downtown Dubai"), (25.0657, 55.1713, "Jumeirah"),
        (24.4667, 54.3667, "Al Khalidiya"), (24.2992, 54.6973, "Mussafah"),
        (24.2075, 55.7447, "Al Ain"), (24.1302, 52.7146, "Madinat Zayed"),
        (25.4052, 55.5136, "Ajman Industrial"), (25.2911, 51.5304, "Al Ruwais")
    ],
    "GB": [
        (51.5074, -0.1278, "London"), (53.4808, -2.2426, "Manchester"),
        (52.4862, -1.8904, "Birmingham"), (55.9533, -3.1883, "Edinburgh"),
        (51.4545, -2.5879, "Bristol")
    ]
}

NO_IMAGERY_HASH = "950bfc22ebfe0f2e4879e640d980820b64be5a91b167ff9d577510ec17e587d9"

def is_no_imagery_image(content):
    try:
        image = Image.open(BytesIO(content))
        sha = hashlib.sha256(content).hexdigest()
        return sha == NO_IMAGERY_HASH
    except Exception:
        return True

def get_country_code(lat, lon, api_key):
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"latlng": f"{lat},{lon}", "key": api_key}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        results = response.json().get("results", [])
        for component in results[0]["address_components"]:
            if "country" in component["types"]:
                return component["short_name"], component["long_name"]
    return None, None

def generate_random_coords_near_urban_centers(country_code, count):
    if country_code not in urban_centers:
        print(f"❌ No urban centers defined for: {country_code}")
        return []

    coords = []
    while len(coords) < count:
        city_lat, city_lon, city_name = random.choice(urban_centers[country_code])
        offset_range = 0.01 if country_code == "AE" else 0.03
        delta_lat = random.uniform(-offset_range, offset_range)
        delta_lon = random.uniform(-offset_range, offset_range)
        coords.append((city_lat + delta_lat, city_lon + delta_lon, city_name))
    return coords

def download_image_with_retries(lat, lon, heading, api_key, folder, max_retries=3):
    base_url = "https://maps.googleapis.com/maps/api/streetview"
    params = {
        "size": "1024x1024",
        "location": f"{lat},{lon}",
        "fov": "90",
        "heading": heading,
        "pitch": "0",
        "key": api_key
    }
    filename = os.path.join(folder, f"{lat:.6f}_{lon:.6f}_{heading}.jpg")
    if os.path.exists(filename):
        return False, None

    for attempt in range(max_retries):
        try:
            response = requests.get(base_url, params=params, timeout=10)
            if response.status_code == 200 and len(response.content) > 1000:
                if is_no_imagery_image(response.content):
                    print(f"🚫 No imagery at {lat},{lon} heading {heading}")
                    return False, None
                with open(filename, 'wb') as f:
                    f.write(response.content)
                print(f"✔️ Saved: {filename}")
                return True, filename
        except requests.RequestException as e:
            print(f"⚠️ Retry {attempt + 1} failed for {filename}: {e}")
        time.sleep(0.5)
    print(f"❌ Failed permanently: {filename}")
    return False, None

def download_dataset_from_country_center(lat, lon, api_key, base_folder, target_images=15000, max_workers=10):
    short_code, long_name = get_country_code(lat, lon, api_key)
    if not short_code:
        print("❌ Could not determine country from coordinates.")
        return

    print(f"\n🌍 Country detected: {long_name} ({short_code})\n")
    coords = generate_random_coords_near_urban_centers(short_code, target_images * 2)
    headings = [0, 90, 180, 270]
    saved_images = 0
    futures = []

    city_folders = {}
    for city in set(c for _, _, c in coords):
        city_folder = os.path.join(base_folder, long_name, city)
        os.makedirs(city_folder, exist_ok=True)
        metadata_file = open(os.path.join(city_folder, "metadata.csv"), "a", newline="", encoding="utf-8")
        writer = csv.DictWriter(metadata_file, fieldnames=["filename", "lat", "lon", "heading"])
        if os.stat(metadata_file.name).st_size == 0:
            writer.writeheader()
        city_folders[city] = {"folder": city_folder, "metadata": metadata_file, "writer": writer}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for lat, lon, city in coords:
            if saved_images >= target_images:
                break
            for heading in headings:
                if saved_images >= target_images:
                    break
                city_folder = city_folders[city]["folder"]
                futures.append(executor.submit(download_image_with_retries, lat, lon, heading, api_key, city_folder))

        for future in as_completed(futures):
            success, filename = future.result()
            if success and filename:
                basename = os.path.basename(filename)
                parts = basename.replace(".jpg", "").split("_")
                city = os.path.basename(os.path.dirname(filename))
                city_folders[city]["writer"].writerow({
                    "filename": basename,
                    "lat": parts[0],
                    "lon": parts[1],
                    "heading": parts[2]
                })
                saved_images += 1
            if saved_images >= target_images:
                break

    for c in city_folders.values():
        c["metadata"].close()

    print(f"\n✅ Completed: {saved_images} images saved in {long_name} city folders.\n")

# === CONFIGURATION BLOCK ===
if __name__ == "__main__":
    API_KEY = "AIzaSyCjUleAaspoLjVJ3QDk9WQrxBve0MbYycs"
    START_LAT = 51.45944055010345
    START_LON = -1.409715268944014
    OUTPUT_FOLDER = r"D:\DATASET"
    TARGET_IMAGES = 20000

    download_dataset_from_country_center(
        START_LAT,
        START_LON,
        API_KEY,
        OUTPUT_FOLDER,
        target_images=TARGET_IMAGES
    )
