import os
import requests

BASE_URL = "https://ddragon.leagueoflegends.com"
LANGUAGE = "en_US"
BASE_DIR = "assets"  # change if you want another root folder


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
    print("Using Data Dragon version:", version)

    # -------- Champions: list --------
    champ_list_url = f"{BASE_URL}/cdn/{version}/data/{LANGUAGE}/champion.json"
    champ_list = requests.get(champ_list_url).json()["data"]

    champ_keys = list(champ_list.keys())
    champ_ids = [champ_list[k]["id"] for k in champ_keys]
    print(f"Found {len(champ_ids)} champions")

    # -------- Champion assets --------
    for champ_key, champ_id in zip(champ_keys, champ_ids):
        print(f"\nChampion: {champ_id}")

        champ_root = os.path.join(BASE_DIR, "champions", champ_id)
        square_dir = os.path.join(champ_root, "square")
        passive_dir = os.path.join(champ_root, "passives")
        spells_dir = os.path.join(champ_root, "spells")
        loading_dir = os.path.join(champ_root, "loading_screen")

        # Square icon
        icon_url = f"{BASE_URL}/cdn/{version}/img/champion/{champ_id}.png"
        out_path = os.path.join(square_dir, f"{champ_id}.png")
        download_file(icon_url, out_path)

        # Champion detail JSON
        champ_detail_url = (
            f"{BASE_URL}/cdn/{version}/data/{LANGUAGE}/champion/{champ_key}.json"
        )
        champ_detail = requests.get(champ_detail_url).json()["data"][champ_key]

        # Passive icon
        passive = champ_detail.get("passive")
        if passive and "image" in passive:
            passive_filename = passive["image"]["full"]
            passive_url = f"{BASE_URL}/cdn/{version}/img/passive/{passive_filename}"
            out_path = os.path.join(passive_dir, passive_filename)
            download_file(passive_url, out_path)

        # All spell icons (Q, W, E, R)
        for spell in champ_detail.get("spells", []):
            spell_filename = spell["image"]["full"]
            spell_url = f"{BASE_URL}/cdn/{version}/img/spell/{spell_filename}"
            out_path = os.path.join(spells_dir, spell_filename)
            download_file(spell_url, out_path)

        # All skin-specific loading screen assets
        # Uses the `skins` array and each skin's `num` field as documented
        # Aatrox_0.jpg, Aatrox_1.jpg, ...
        for skin in champ_detail.get("skins", []):
            skin_num = skin.get("num", 0)
            loading_url = (
                f"{BASE_URL}/cdn/img/champion/loading/{champ_id}_{skin_num}.jpg"
            )
            out_path = os.path.join(
                loading_dir, f"{champ_id}_{skin_num}.jpg"
            )
            download_file(loading_url, out_path)

    # -------- Items --------
    print("\nDownloading all item icons...")
    item_data_url = f"{BASE_URL}/cdn/{version}/data/{LANGUAGE}/item.json"
    item_data = requests.get(item_data_url).json()["data"]

    item_dir = os.path.join(BASE_DIR, "items")
    print(f"Found {len(item_data)} items")

    for item_id in item_data.keys():
        item_filename = f"{item_id}.png"
        item_url = f"{BASE_URL}/cdn/{version}/img/item/{item_filename}"
        out_path = os.path.join(item_dir, item_filename)
        download_file(item_url, out_path)

    print("\nDone. Check the 'assets/' folder.")


if __name__ == "__main__":
    main()
 