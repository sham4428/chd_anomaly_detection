import torch
from model import ECGAutoencoder, PCGAutoencoder, CXRAutoencoder


def test_ecg():
    print("测试 ECG Autoencoder...")
    model = ECGAutoencoder()
    
    x = torch.randn(2, 12, 1000)
    
    with torch.no_grad():
        recon = model(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {recon.shape}")
    assert recon.shape == x.shape, "ECG输入输出维度不匹配！"
    print("ECG Autoencoder 测试通过！\n")


def test_pcg():
    print("测试 PCG Autoencoder...")
    model = PCGAutoencoder()
    
    x = torch.randn(2, 13, 200)
    
    with torch.no_grad():
        recon = model(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {recon.shape}")
    assert recon.shape == x.shape, "PCG输入输出维度不匹配！"
    print("PCG Autoencoder 测试通过！\n")


def test_cxr():
    print("测试 CXR Autoencoder...")
    model = CXRAutoencoder()
    
    x = torch.randn(2, 1, 256, 256)
    
    with torch.no_grad():
        recon = model(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {recon.shape}")
    assert recon.shape == x.shape, "CXR输入输出维度不匹配！"
    print("CXR Autoencoder 测试通过！\n")


if __name__ == "__main__":
    print("=" * 50)
    print("开始测试 Transformer/ViT 模型")
    print("=" * 50 + "\n")
    
    test_ecg()
    test_pcg()
    test_cxr()
    
    print("=" * 50)
    print("所有模型测试通过！")
    print("=" * 50)
