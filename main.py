import torch

def main():
    print("Hello from zalo-ai!")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.__version__}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")

if __name__ == "__main__":
    main()
