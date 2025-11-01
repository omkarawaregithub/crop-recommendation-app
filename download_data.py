import requests

def download_dataset():
    url = "https://raw.githubusercontent.com/omkarawaregithub/crop-recommendation-app/main/Crop_recommendation.csv"
    response = requests.get(url)
    
    with open("Crop_recommendation.csv", "wb") as f:
        f.write(response.content)
    
    print("Dataset downloaded successfully!")

if __name__ == "__main__":
    download_dataset()