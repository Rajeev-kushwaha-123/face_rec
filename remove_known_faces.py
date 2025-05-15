import pickle
import os

PKL_PATH = "known_faces.pkl"

def load_known_faces():
    if not os.path.exists(PKL_PATH):
        print(f"File {PKL_PATH} does not exist.")
        return None, None
    with open(PKL_PATH, "rb") as f:
        data = pickle.load(f)
        return data["encodings"], data["names"]

def save_known_faces(encodings, names):
    data = {"encodings": encodings, "names": names}
    with open(PKL_PATH, "wb") as f:
        pickle.dump(data, f)
    print(f"Updated data saved to {PKL_PATH}.")

def main():
    encodings, names = load_known_faces()
    if encodings is None or names is None:
        return
    print("Current names in database:")
    for idx, name in enumerate(names):
        print(f"{idx}: {name}")
    to_remove = input("Enter names to remove (comma-separated): ").strip().split(",")
    to_remove = [n.strip() for n in to_remove if n.strip()]
    if not to_remove:
        print("No names specified. Exiting.")
        return
    new_encodings = []
    new_names = []
    for encoding, name in zip(encodings, names):
        if name not in to_remove:
            new_encodings.append(encoding)
            new_names.append(name)
        else:
            print(f"Removing: {name}")
    save_known_faces(new_encodings, new_names)
    print("Done.")

if __name__ == "__main__":
    main() 