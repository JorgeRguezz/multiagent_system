import os
import requests

BASE_URL = "https://ddragon.leagueoflegends.com"
LANGUAGE = "en_US"

def get_latest_version():
    versions_url = f"{BASE_URL}/api/versions.json"
    resp = requests.get(versions_url)
    resp.raise_for_status()
    return resp.json()[0]

def download_file(url, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    resp = requests.get(url, stream=True)
    if resp.status_code == 200:
        with open(path, "wb") as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)
        print(f"OK  {path}")
    else:
        print(f"SKIP {path} (status {resp.status_code})")

def main():
    version = get_latest_version()
    print("Using version:", version)

    base_dir = "assets_test"

    # ---- Load champion list ----
    champ_list_url = f"{BASE_URL}/cdn/{version}/data/{LANGUAGE}/champion.json"
    champ_list = requests.get(champ_list_url).json()["data"]

    # Take only 2 champions for the test
    champ_keys = list(champ_list.keys())[:2]
    champ_ids = [champ_list[k]["id"] for k in champ_keys]
    print("Test champions:", champ_ids)

    # Champion assets per-champion folder
    for champ_key, champ_id in zip(champ_keys, champ_ids):
        champ_root = os.path.join(base_dir, "champions", champ_id)
        square_dir = os.path.join(champ_root, "square")
        passive_dir = os.path.join(champ_root, "passives")
        spells_dir = os.path.join(champ_root, "spells")

        # Champion square
        icon_url = f"{BASE_URL}/cdn/{version}/img/champion/{champ_id}.png"
        out_path = os.path.join(square_dir, f"{champ_id}.png")
        download_file(icon_url, out_path)

        # Champion details
        champ_detail_url = f"{BASE_URL}/cdn/{version}/data/{LANGUAGE}/champion/{champ_key}.json"
        champ_detail = requests.get(champ_detail_url).json()["data"][champ_key]

        # Passive
        passive = champ_detail.get("passive")
        if passive and "image" in passive:
            passive_filename = passive["image"]["full"]   # e.g. Anivia_P.png
            passive_url = f"{BASE_URL}/cdn/{version}/img/passive/{passive_filename}"
            out_path = os.path.join(passive_dir, passive_filename)
            download_file(passive_url, out_path)

        # ALL spells (Q, W, E, R) instead of only the first one
        for spell in champ_detail.get("spells", []):
            spell_filename = spell["image"]["full"]       # e.g. FlashFrost.png
            spell_url = f"{BASE_URL}/cdn/{version}/img/spell/{spell_filename}"
            out_path = os.path.join(spells_dir, spell_filename)
            download_file(spell_url, out_path)

    # ---- A few items (unchanged layout) ----
    item_data_url = f"{BASE_URL}/cdn/{version}/data/{LANGUAGE}/item.json"
    item_data = requests.get(item_data_url).json()["data"]

    # Take 5 items
    item_ids = list(item_data.keys())[:5]
    print("Test items:", item_ids)

    for item_id in item_ids:
        item_filename = f"{item_id}.png"
        item_url = f"{BASE_URL}/cdn/{version}/img/item/{item_filename}"
        out_path = os.path.join(base_dir, "items", item_filename)
        download_file(item_url, out_path)

    print("\nTest download complete. Check the assets_test/ folder.")

if __name__ == "__main__":
    main()
 