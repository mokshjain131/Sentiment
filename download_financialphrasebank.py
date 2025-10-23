"""
Download FinancialPhraseBank from Hugging Face and convert to your format.

Usage:
    python download_financialphrasebank.py --format csv
    python download_financialphrasebank.py --format parquet
"""
import pandas as pd
import argparse
from pathlib import Path

def download_and_convert(output_format: str = "parquet"):
    """
    Downloads FinancialPhraseBank and converts to your project's format.
    Uses the mltrev23/financial-sentiment-analysis dataset from Hugging Face.
    """
    print("📥 Downloading FinancialPhraseBank from Hugging Face (mltrev23 version)...")
    
    try:
        from datasets import load_dataset
        
        # Use the working version from mltrev23
        dataset = load_dataset("mltrev23/financial-sentiment-analysis")
        
        print(f"✅ Downloaded {len(dataset['train'])} sentences")
        
        # Convert to pandas
        df = dataset['train'].to_pandas()
        
        # Check what columns we have
        print(f"   Columns: {df.columns.tolist()}")
        
        # Standardize column names (mltrev23 uses 'Sentence' and 'Sentiment')
        column_mapping = {
            'Sentence': 'text',
            'sentence': 'text',
            'Sentiment': 'label',
            'sentiment': 'label'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns and new_col not in df.columns:
                df = df.rename(columns={old_col: new_col})
        
        # The mltrev23 dataset has 'text' and 'label' columns already
        # Labels might be strings or integers - let's standardize
        if 'label' in df.columns:
            # If numeric, map to strings
            if df['label'].dtype in ['int64', 'int32']:
                label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
                df['label'] = df['label'].map(label_map)
            # Standardize label casing
            df['label'] = df['label'].str.lower()
        
        # Ensure we have text column
        if 'text' not in df.columns and 'sentence' in df.columns:
            df = df.rename(columns={'sentence': 'text'})
        
        # Add metadata
        df['source'] = 'FinancialPhraseBank'
        df['dataset_type'] = 'gold_standard'
        
    except ImportError:
        print("❌ 'datasets' library not found. Installing...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'datasets'])
        print("✅ Installed. Please run this script again.")
        return None
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("\n💡 Alternative: Download manually from Kaggle:")
        print("https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news")
        return None
    
    # Save to data/datasets directory
    output_dir = Path("data/datasets")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if output_format.lower() == "csv":
        output_path = output_dir / "financialphrasebank.csv"
        df.to_csv(output_path, index=False)
    else:
        output_path = output_dir / "financialphrasebank.parquet"
        df.to_parquet(output_path, index=False)
    
    print(f"\n✅ Saved to: {output_path}")
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total sentences: {len(df)}")
    print(f"\n   Label distribution:")
    print(df['label'].value_counts().to_string())
    print(f"\n🎯 Next steps:")
    print(f"   1. Train your model: python -m src.cli.train_model --dataset {output_path} --model ProsusAI/finbert --out models/finetuned_fpb")
    print(f"   2. The model will be saved to: models/finetuned_fpb/")
    
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download FinancialPhraseBank dataset")
    parser.add_argument("--format", choices=["csv", "parquet"], default="parquet", 
                       help="Output format (default: parquet)")
    args = parser.parse_args()
    
    download_and_convert(args.format)
