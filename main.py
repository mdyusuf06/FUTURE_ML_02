from preprocess import load_data
from train_model import train_models
from predict import predict_ticket

def main():

    print("Loading dataset...")
    df = load_data("tickets.csv")

    vectorizer, cat_model, pri_model = train_models(df)

    print("\nSystem ready!\n")

    while True:
        text = input("Enter a support ticket (or type 'exit'): ")

        if text.lower() == "exit":
            print("Exiting...")
            break

        predict_ticket(text)

if __name__ == "__main__":
    main()
