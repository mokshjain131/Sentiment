"""
Financial Sentiment Analysis - Desktop Application
Modern GUI interface using tkinter with customtkinter for enhanced visuals
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import sys
from pathlib import Path
from datetime import datetime, timedelta
import threading
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.loaders.news_api_loader import fetch_and_store
from src.input.schemas import FetchParams
from src.aggregation.report_builder import build_reports
from src.services.news_api_client import load_config


class SentimentAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Financial Sentiment Analysis")
        self.root.geometry("1000x700")
        self.root.minsize(900, 600)
        
        # Color scheme - Professional dark theme with high contrast
        self.colors = {
            'bg_dark': '#1e1e1e',
            'bg_medium': '#2d2d2d',
            'bg_light': '#3d3d3d',
            'accent': '#007acc',
            'accent_hover': '#005a9e',
            'success': '#4caf50',
            'warning': '#ff9800',
            'error': '#f44336',
            'text': '#ffffff',
            'text_secondary': '#e0e0e0',  # Much lighter for better contrast on dark bg
            'text_muted': '#888888',      # Darker for contrast on light bg
            'border': '#555555'
        }
        
        # Configure root
        self.root.configure(bg=self.colors['bg_dark'])
        
        # State variables
        self.dataset_path = tk.StringVar()
        self.ticker_var = tk.StringVar(value="AAPL")
        self.is_processing = False
        
        # Setup UI
        self.setup_ui()
        
        # Check initial setup
        self.check_setup()
    
    def setup_ui(self):
        """Create the main UI layout"""
        
        # Header
        header_frame = tk.Frame(self.root, bg=self.colors['bg_medium'], height=80)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(
            header_frame,
            text=" Financial Sentiment Analysis",
            font=("Segoe UI", 20, "bold"),
            bg=self.colors['bg_medium'],
            fg=self.colors['text']
        )
        title_label.pack(pady=10)
        
        subtitle_label = tk.Label(
            header_frame,
            text="FinBERT Fine-Tuned Model ¢ 81.8% Accuracy ¢ Production Ready",
            font=("Segoe UI", 10),
            bg=self.colors['bg_medium'],
            fg=self.colors['text_secondary']
        )
        subtitle_label.pack()
        
        # Main content area with notebook (tabs)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Configure notebook style
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TNotebook', background=self.colors['bg_dark'], borderwidth=0)
        style.configure('TNotebook.Tab', background=self.colors['bg_medium'], 
                       foreground=self.colors['text'], padding=[20, 10])
        style.map('TNotebook.Tab', background=[('selected', self.colors['accent'])])
        
        # Create tabs
        self.create_fetch_tab()
        self.create_analyze_tab()
        self.create_test_tab()
        self.create_help_tab()
        
        # Status bar
        self.status_bar = tk.Label(
            self.root,
            text="Ready",
            font=("Segoe UI", 9),
            bg=self.colors['bg_medium'],
            fg=self.colors['text_secondary'],
            anchor=tk.W,
            padx=10
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def create_fetch_tab(self):
        """Tab 1: Fetch News"""
        tab = tk.Frame(self.notebook, bg=self.colors['bg_dark'])
        self.notebook.add(tab, text="° Fetch News")
        
        # Scrollable frame
        canvas = tk.Canvas(tab, bg=self.colors['bg_dark'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.colors['bg_dark'])
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Instructions
        instructions = tk.Label(
            scrollable_frame,
            text="Enter company details to fetch financial news articles from NewsAPI",
            font=("Segoe UI", 11),
            bg=self.colors['bg_dark'],
            fg=self.colors['text_secondary'],
            wraplength=850,
            justify=tk.LEFT
        )
        instructions.pack(pady=20, padx=20, anchor=tk.W)
        
        # Input frame
        input_frame = tk.Frame(scrollable_frame, bg=self.colors['bg_dark'])
        input_frame.pack(fill=tk.BOTH, padx=20, pady=10)
        
        # Company details
        self.create_labeled_entry(input_frame, "Company Name:", "company_name", "Apple Inc", 0)
        self.create_labeled_entry(input_frame, "Ticker Symbol:", "ticker_fetch", "AAPL", 1)
        self.create_labeled_entry(input_frame, "Keywords (comma-separated):", "keywords", "iPhone,services,earnings", 2)
        
        # Settings
        self.create_labeled_scale(input_frame, "Days to Fetch:", "days", 1, 30, 7, 3)
        self.create_labeled_scale(input_frame, "Max Articles:", "max_articles", 50, 500, 150, 4)
        
        # Format selection
        format_frame = tk.Frame(input_frame, bg=self.colors['bg_dark'])
        format_frame.grid(row=5, column=0, columnspan=2, pady=10, sticky=tk.W)
        
        tk.Label(
            format_frame,
            text="Output Format:",
            font=("Segoe UI", 10),
            bg=self.colors['bg_dark'],
            fg=self.colors['text']
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        self.format_var = tk.StringVar(value="csv")
        for fmt in ["csv", "parquet"]:
            tk.Radiobutton(
                format_frame,
                text=fmt.upper(),
                variable=self.format_var,
                value=fmt,
                font=("Segoe UI", 10),
                bg=self.colors['bg_dark'],
                fg=self.colors['text'],
                selectcolor=self.colors['bg_medium'],
                activebackground=self.colors['bg_dark'],
                activeforeground=self.colors['text']
            ).pack(side=tk.LEFT, padx=5)
        
        # Fetch button
        fetch_btn = tk.Button(
            scrollable_frame,
            text=" Fetch News Articles",
            font=("Segoe UI", 12, "bold"),
            bg=self.colors['accent'],
            fg=self.colors['text'],
            activebackground=self.colors['accent_hover'],
            activeforeground=self.colors['text'],
            relief=tk.FLAT,
            cursor="hand2",
            command=self.fetch_news,
            padx=30,
            pady=15
        )
        fetch_btn.pack(pady=20)
        
        # Output area
        output_label = tk.Label(
            scrollable_frame,
            text="Results:",
            font=("Segoe UI", 10, "bold"),
            bg=self.colors['bg_dark'],
            fg=self.colors['text'],
            anchor=tk.W
        )
        output_label.pack(pady=(20, 5), padx=20, anchor=tk.W)
        
        self.fetch_output = scrolledtext.ScrolledText(
            scrollable_frame,
            height=12,
            font=("Consolas", 9),
            bg=self.colors['bg_medium'],
            fg=self.colors['text'],
            insertbackground=self.colors['text'],
            relief=tk.FLAT,
            padx=10,
            pady=10
        )
        self.fetch_output.pack(fill=tk.BOTH, padx=20, pady=(0, 20), expand=True)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def create_analyze_tab(self):
        """Tab 2: Analyze Sentiment"""
        tab = tk.Frame(self.notebook, bg=self.colors['bg_dark'])
        self.notebook.add(tab, text=" Analyze Sentiment")
        
        # Scrollable frame
        canvas = tk.Canvas(tab, bg=self.colors['bg_dark'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.colors['bg_dark'])
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Instructions
        instructions = tk.Label(
            scrollable_frame,
            text="Analyze sentiment from fetched news articles and generate reports",
            font=("Segoe UI", 11),
            bg=self.colors['bg_dark'],
            fg=self.colors['text_secondary'],
            wraplength=850,
            justify=tk.LEFT
        )
        instructions.pack(pady=20, padx=20, anchor=tk.W)
        
        # Input frame
        input_frame = tk.Frame(scrollable_frame, bg=self.colors['bg_dark'])
        input_frame.pack(fill=tk.BOTH, padx=20, pady=10)
        
        # Dataset path with browse button
        path_frame = tk.Frame(input_frame, bg=self.colors['bg_dark'])
        path_frame.grid(row=0, column=0, columnspan=2, sticky=tk.EW, pady=10)
        
        tk.Label(
            path_frame,
            text="Dataset Path:",
            font=("Segoe UI", 10),
            bg=self.colors['bg_dark'],
            fg=self.colors['text'],
            anchor=tk.W
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        self.dataset_entry = tk.Entry(
            path_frame,
            textvariable=self.dataset_path,
            font=("Segoe UI", 10),
            bg=self.colors['bg_medium'],
            fg=self.colors['text'],
            insertbackground=self.colors['text'],
            relief=tk.FLAT,
            width=50
        )
        self.dataset_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=5, padx=(0, 10))
        
        browse_btn = tk.Button(
            path_frame,
            text="Browse...",
            font=("Segoe UI", 9),
            bg=self.colors['bg_light'],
            fg=self.colors['text'],
            activebackground=self.colors['bg_medium'],
            relief=tk.FLAT,
            cursor="hand2",
            command=self.browse_dataset,
            padx=15,
            pady=5
        )
        browse_btn.pack(side=tk.LEFT)
        
        # Other inputs
        self.create_labeled_entry(input_frame, "Ticker Symbol:", "ticker_analyze", "AAPL", 1)
        self.create_labeled_scale(input_frame, "Rolling Window (days):", "window", 3, 30, 7, 2)
        self.create_labeled_scale(input_frame, "Alert Threshold (z-score):", "threshold", 1.0, 5.0, 2.5, 3, resolution=0.1)
        
        # Format selection
        format_frame = tk.Frame(input_frame, bg=self.colors['bg_dark'])
        format_frame.grid(row=4, column=0, columnspan=2, pady=10, sticky=tk.W)
        
        tk.Label(
            format_frame,
            text="Output Format:",
            font=("Segoe UI", 10),
            bg=self.colors['bg_dark'],
            fg=self.colors['text']
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        self.format_analyze_var = tk.StringVar(value="csv")
        for fmt in ["csv", "parquet"]:
            tk.Radiobutton(
                format_frame,
                text=fmt.upper(),
                variable=self.format_analyze_var,
                value=fmt,
                font=("Segoe UI", 10),
                bg=self.colors['bg_dark'],
                fg=self.colors['text'],
                selectcolor=self.colors['bg_medium'],
                activebackground=self.colors['bg_dark'],
                activeforeground=self.colors['text']
            ).pack(side=tk.LEFT, padx=5)
        
        # Analyze button
        analyze_btn = tk.Button(
            scrollable_frame,
            text=" Analyze Sentiment",
            font=("Segoe UI", 12, "bold"),
            bg=self.colors['success'],
            fg=self.colors['text'],
            activebackground='#45a049',
            activeforeground=self.colors['text'],
            relief=tk.FLAT,
            cursor="hand2",
            command=self.analyze_sentiment,
            padx=30,
            pady=15
        )
        analyze_btn.pack(pady=20)
        
        # Output area
        output_label = tk.Label(
            scrollable_frame,
            text="Analysis Results:",
            font=("Segoe UI", 10, "bold"),
            bg=self.colors['bg_dark'],
            fg=self.colors['text'],
            anchor=tk.W
        )
        output_label.pack(pady=(20, 5), padx=20, anchor=tk.W)
        
        self.analyze_output = scrolledtext.ScrolledText(
            scrollable_frame,
            height=12,
            font=("Consolas", 9),
            bg=self.colors['bg_medium'],
            fg=self.colors['text'],
            insertbackground=self.colors['text'],
            relief=tk.FLAT,
            padx=10,
            pady=10
        )
        self.analyze_output.pack(fill=tk.BOTH, padx=20, pady=(0, 20), expand=True)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def create_test_tab(self):
        """Tab 3: Test Sentence"""
        tab = tk.Frame(self.notebook, bg=self.colors['bg_dark'])
        self.notebook.add(tab, text="§ª Test Sentence")
        
        # Instructions
        instructions = tk.Label(
            tab,
            text="Test the sentiment model on any financial sentence",
            font=("Segoe UI", 11),
            bg=self.colors['bg_dark'],
            fg=self.colors['text_secondary']
        )
        instructions.pack(pady=20, padx=20)
        
        # Input area
        input_label = tk.Label(
            tab,
            text="Enter Financial Text:",
            font=("Segoe UI", 10, "bold"),
            bg=self.colors['bg_dark'],
            fg=self.colors['text'],
            anchor=tk.W
        )
        input_label.pack(pady=(20, 5), padx=20, anchor=tk.W)
        
        self.test_input = scrolledtext.ScrolledText(
            tab,
            height=4,
            font=("Segoe UI", 10),
            bg=self.colors['bg_medium'],
            fg=self.colors['text'],
            insertbackground=self.colors['text'],
            relief=tk.FLAT,
            padx=10,
            pady=10,
            wrap=tk.WORD
        )
        self.test_input.pack(fill=tk.X, padx=20, pady=(0, 10))
        
        # Example buttons
        example_frame = tk.Frame(tab, bg=self.colors['bg_dark'])
        example_frame.pack(pady=10)
        
        examples = [
            (" Positive Example", "Company profits surged 50% exceeding all expectations"),
            (" Neutral Example", "The firm reported quarterly earnings as expected"),
            (" Negative Example", "Stock plunged amid scandal and massive layoffs")
        ]
        
        for text, example in examples:
            btn = tk.Button(
                example_frame,
                text=text,
                font=("Segoe UI", 9),
                bg=self.colors['bg_light'],
                fg=self.colors['text'],
                activebackground=self.colors['bg_medium'],
                relief=tk.FLAT,
                cursor="hand2",
                command=lambda e=example: self.set_example(e),
                padx=15,
                pady=8
            )
            btn.pack(side=tk.LEFT, padx=5)
        
        # Predict button
        predict_btn = tk.Button(
            tab,
            text=" Predict Sentiment",
            font=("Segoe UI", 12, "bold"),
            bg=self.colors['accent'],
            fg=self.colors['text'],
            activebackground=self.colors['accent_hover'],
            activeforeground=self.colors['text'],
            relief=tk.FLAT,
            cursor="hand2",
            command=self.predict_sentiment,
            padx=30,
            pady=15
        )
        predict_btn.pack(pady=20)
        
        # Output area
        output_label = tk.Label(
            tab,
            text="Prediction:",
            font=("Segoe UI", 10, "bold"),
            bg=self.colors['bg_dark'],
            fg=self.colors['text'],
            anchor=tk.W
        )
        output_label.pack(pady=(20, 5), padx=20, anchor=tk.W)
        
        self.test_output = scrolledtext.ScrolledText(
            tab,
            height=10,
            font=("Consolas", 10),
            bg=self.colors['bg_medium'],
            fg=self.colors['text'],
            insertbackground=self.colors['text'],
            relief=tk.FLAT,
            padx=10,
            pady=10
        )
        self.test_output.pack(fill=tk.BOTH, padx=20, pady=(0, 20), expand=True)
    
    def create_help_tab(self):
        """Tab 4: Help"""
        tab = tk.Frame(self.notebook, bg=self.colors['bg_dark'])
        self.notebook.add(tab, text=" Help")
        
        help_text = scrolledtext.ScrolledText(
            tab,
            font=("Segoe UI", 10),
            bg=self.colors['bg_medium'],
            fg=self.colors['text'],
            insertbackground=self.colors['text'],
            relief=tk.FLAT,
            padx=15,
            pady=15,
            wrap=tk.WORD
        )
        help_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        help_content = """
 USER GUIDE

SETUP:
1. Set NewsAPI key: $env:NEWSAPI_KEY="your_key"
2. Ensure model is at: models/Colab/FinBERT_improved/
3. Run: python desktop_app.py

USAGE FLOW:

° Tab 1: Fetch News
   ¢ Enter company name and ticker
   ¢ Add keywords (comma-separated)
   ¢ Choose days to fetch and max articles
   ¢ Click "Fetch News"
   ¢ Dataset path will auto-fill for next step

 Tab 2: Analyze Sentiment
   ¢ Dataset path should be auto-filled
   ¢ Adjust rolling window (default: 7 days)
   ¢ Adjust alert threshold (default: 2.5)
   ¢ Click "Analyze Sentiment"
   ¢ View daily metrics and alerts

§ª Tab 3: Test Sentences
   ¢ Enter any financial text
   ¢ Or click example buttons
   ¢ Click "Predict Sentiment"
   ¢ See sentiment probabilities

OUTPUT FILES:
¢ Fetched news: data/processed/YYYYMMDD/ticker_YYYYMMDD.{csv|parquet}
¢ Daily metrics: reports/ticker/daily.{csv|parquet}
¢ Alerts: reports/ticker/alerts.{csv|parquet}

MODEL DETAILS:
¢ Base: ProsusAI/finbert
¢ Training: FinancialPhraseBank (5,842 sentences)
¢ Performance: 81.8% accuracy, 79.3% F1 macro
¢ Classes: Negative, Neutral, Positive

TROUBLESHOOTING:

"NEWSAPI_KEY not set"
 Set environment variable: $env:NEWSAPI_KEY="key"

"Model not found"
 Ensure model at: models/Colab/FinBERT_improved/

"No articles fetched"
 Check API key, try different keywords

KEYBOARD SHORTCUTS:
¢ Ctrl+C: Copy from output areas
¢ Ctrl+A: Select all in input areas
¢ Ctrl+V: Paste in input areas

DOCUMENTATION:
¢ README.md - Quick start guide
¢ COMPLETE_DOCUMENTATION.md - Full documentation
¢ UI_GUIDE.md - Web UI documentation

STATUS INDICATORS:
¢  Success (green message)
¢  Warning (yellow message)
¢  Error (red message)
¢  Processing (blue message)

For complete documentation, see COMPLETE_DOCUMENTATION.md Section 11.
        """
        
        help_text.insert(1.0, help_content)
        help_text.config(state=tk.DISABLED)
    
    def create_labeled_entry(self, parent, label, var_name, default, row):
        """Helper to create labeled entry field"""
        tk.Label(
            parent,
            text=label,
            font=("Segoe UI", 10),
            bg=self.colors['bg_dark'],
            fg=self.colors['text'],
            anchor=tk.W
        ).grid(row=row, column=0, sticky=tk.W, pady=10)
        
        var = tk.StringVar(value=default)
        setattr(self, var_name, var)
        
        entry = tk.Entry(
            parent,
            textvariable=var,
            font=("Segoe UI", 10),
            bg=self.colors['bg_medium'],
            fg=self.colors['text'],
            insertbackground=self.colors['text'],
            relief=tk.FLAT,
            width=40
        )
        entry.grid(row=row, column=1, sticky=tk.EW, pady=10, padx=(10, 0), ipady=5)
        
        parent.columnconfigure(1, weight=1)
    
    def create_labeled_scale(self, parent, label, var_name, from_, to, default, row, resolution=1):
        """Helper to create labeled scale"""
        tk.Label(
            parent,
            text=label,
            font=("Segoe UI", 10),
            bg=self.colors['bg_dark'],
            fg=self.colors['text'],
            anchor=tk.W
        ).grid(row=row, column=0, sticky=tk.W, pady=10)
        
        frame = tk.Frame(parent, bg=self.colors['bg_dark'])
        frame.grid(row=row, column=1, sticky=tk.EW, pady=10, padx=(10, 0))
        
        var = tk.DoubleVar(value=default)
        setattr(self, var_name, var)
        
        scale = tk.Scale(
            frame,
            variable=var,
            from_=from_,
            to=to,
            resolution=resolution,
            orient=tk.HORIZONTAL,
            font=("Segoe UI", 9),
            bg=self.colors['bg_medium'],
            fg=self.colors['text'],
            activebackground=self.colors['accent'],
            troughcolor=self.colors['bg_light'],
            highlightthickness=0,
            relief=tk.FLAT,
            length=300
        )
        scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        value_label = tk.Label(
            frame,
            textvariable=var,
            font=("Segoe UI", 10, "bold"),
            bg=self.colors['bg_dark'],
            fg=self.colors['accent'],
            width=8
        )
        value_label.pack(side=tk.LEFT, padx=(10, 0))
    
    def set_status(self, message, color='text_secondary'):
        """Update status bar"""
        self.status_bar.config(text=message, fg=self.colors[color])
        self.root.update()
    
    def check_setup(self):
        """Check initial setup"""
        # Check API key
        try:
            config = load_config()
            if not config.newsapi_key:
                self.set_status(" Warning: NEWSAPI_KEY not set", 'warning')
            else:
                self.set_status(" Ready - API key configured", 'success')
        except:
            self.set_status(" Warning: Could not load config", 'warning')
        
        # Check model
        model_path = Path("models/Colab/FinBERT_improved")
        if not model_path.exists():
            messagebox.showwarning(
                "Model Not Found",
                "Model not found at: models/Colab/FinBERT_improved/\n\n"
                "Prediction features will not work until the model is available."
            )
    
    def browse_dataset(self):
        """Browse for dataset file"""
        filename = filedialog.askopenfilename(
            title="Select Dataset File",
            filetypes=[
                ("CSV files", "*.csv"),
                ("Parquet files", "*.parquet"),
                ("All files", "*.*")
            ],
            initialdir="data/processed"
        )
        if filename:
            self.dataset_path.set(filename)
    
    def set_example(self, text):
        """Set example text"""
        self.test_input.delete(1.0, tk.END)
        self.test_input.insert(1.0, text)
    
    def fetch_news(self):
        """Fetch news articles"""
        if self.is_processing:
            messagebox.showwarning("Processing", "Already processing. Please wait...")
            return
        
        def task():
            self.is_processing = True
            self.set_status(" Fetching news articles...", 'accent')
            
            try:
                # Validate API key
                try:
                    config = load_config()
                    if not config.newsapi_key:
                        self.show_output(self.fetch_output, " Error: NEWSAPI_KEY not set\n", 'error')
                        return
                except Exception as e:
                    self.show_output(self.fetch_output, f" Error: {str(e)}\n", 'error')
                    return
                
                # Get inputs
                company = self.company_name.get()
                ticker = self.ticker_fetch.get()
                keywords_str = self.keywords.get()
                days = int(self.days.get())
                max_articles = int(self.max_articles.get())
                format_type = self.format_var.get()
                
                keywords_list = [k.strip() for k in keywords_str.split(',')] if keywords_str else []
                
                # Calculate dates
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                
                # Create params
                params = FetchParams(
                    company=company,
                    ticker=ticker,
                    keywords=keywords_list,
                    from_date=start_date.strftime('%Y-%m-%d'),
                    to_date=end_date.strftime('%Y-%m-%d'),
                    language='en',
                    max_articles=max_articles
                )
                
                # Fetch
                self.show_output(self.fetch_output, f"Fetching articles for {ticker}...\n", 'accent')
                count = fetch_and_store(params, file_format=format_type)
                
                # Find output file (fetch_and_store uses the article date, not today's date)
                data_dir = Path("data/processed")
                file_ext = 'csv' if format_type == 'csv' else 'parquet'
                
                # Look for the most recent file matching the ticker
                output_file = None
                if data_dir.exists():
                    for date_folder in sorted(data_dir.iterdir(), reverse=True):
                        if date_folder.is_dir():
                            potential_file = date_folder / f"{ticker.lower()}_{date_folder.name}.{file_ext}"
                            if potential_file.exists():
                                output_file = potential_file
                                break
                
                if output_file and output_file.exists():
                    # Load preview
                    if format_type == 'csv':
                        df = pd.read_csv(output_file)
                    else:
                        df = pd.read_parquet(output_file)
                    
                    preview = df[['title', 'source', 'published_at']].head(5)
                    
                    result = f"""
 Success! Fetched {count} articles for {ticker}

Output File: {output_file}

Preview (first 5 articles):
{preview.to_string(index=False)}

Next Step: Go to "Analyze Sentiment" tab to generate reports.
                    """
                    
                    self.show_output(self.fetch_output, result, 'success')
                    self.dataset_path.set(str(output_file))
                    self.ticker_analyze.set(ticker)
                    self.set_status(f" Fetched {count} articles", 'success')
                else:
                    self.show_output(self.fetch_output, f" Fetched {count} articles but file not found at {data_dir}\n", 'warning')
                    self.set_status(f" File not found", 'warning')
                    
            except Exception as e:
                self.show_output(self.fetch_output, f" Error: {str(e)}\n", 'error')
                self.set_status(" Fetch failed", 'error')
            finally:
                self.is_processing = False
        
        thread = threading.Thread(target=task, daemon=True)
        thread.start()
    
    def analyze_sentiment(self):
        """Analyze sentiment"""
        if self.is_processing:
            messagebox.showwarning("Processing", "Already processing. Please wait...")
            return
        
        def task():
            self.is_processing = True
            self.set_status(" Analyzing sentiment...", 'accent')
            
            try:
                dataset_path = self.dataset_path.get()
                if not dataset_path or not Path(dataset_path).exists():
                    self.show_output(self.analyze_output, " Error: Please fetch news first or provide valid dataset path\n", 'error')
                    return
                
                ticker = self.ticker_analyze.get()
                window = int(self.window.get())
                threshold = float(self.threshold.get())
                format_type = self.format_analyze_var.get()
                
                self.show_output(self.analyze_output, f"Step 1: Loading dataset...\n", 'accent')
                
                # Load dataset
                if dataset_path.endswith('.csv'):
                    df = pd.read_csv(dataset_path)
                else:
                    df = pd.read_parquet(dataset_path)
                
                # Ensure tickers column is populated
                if 'tickers' in df.columns:
                    # Parse string representation if needed
                    import ast
                    if df['tickers'].dtype == 'object' and isinstance(df['tickers'].iloc[0], str):
                        df['tickers'] = df['tickers'].apply(lambda x: ast.literal_eval(x) if x and x.strip() else [])
                    
                    # Add ticker to all articles that don't have it
                    df['tickers'] = df['tickers'].apply(lambda x: [ticker.upper()] if not x or len(x) == 0 else x)
                else:
                    # Create tickers column if missing
                    df['tickers'] = [[ticker.upper()]] * len(df)
                
                self.show_output(self.analyze_output, f"Loaded {len(df)} articles for {ticker}\n", 'accent')
                
                # Check if already labeled
                if 'weak_label' not in df.columns:
                    self.show_output(self.analyze_output, f"Step 2: Running sentiment predictions on {len(df)} articles...\n", 'accent')
                    
                    # Load model
                    from transformers import AutoTokenizer, AutoModelForSequenceClassification
                    import torch
                    
                    model_path = "models/Colab/FinBERT_improved"
                    if not Path(model_path).exists():
                        self.show_output(self.analyze_output, " Error: Model not found at models/Colab/FinBERT_improved/\n", 'error')
                        return
                    
                    tokenizer = AutoTokenizer.from_pretrained(model_path)
                    model = AutoModelForSequenceClassification.from_pretrained(model_path)
                    model.eval()
                    
                    # Predict sentiment for each article
                    labels = ["negative", "neutral", "positive"]
                    predictions = []
                    
                    for idx, row in df.iterrows():
                        # Combine title and description for prediction
                        text = f"{row.get('title', '')} {row.get('description', '')}".strip()
                        if not text:
                            predictions.append('neutral')
                            continue
                        
                        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                        with torch.no_grad():
                            outputs = model(**inputs)
                            prediction = outputs.logits.argmax(dim=-1).item()
                        
                        predictions.append(labels[prediction])
                        
                        # Progress update every 10 articles
                        if (idx + 1) % 10 == 0:
                            self.show_output(self.analyze_output, f"Processed {idx + 1}/{len(df)} articles...\n", 'accent')
                    
                    # Add predictions to dataframe
                    df['weak_label'] = predictions
                    
                    # Save labeled dataset
                    labeled_path = dataset_path.replace('.csv', '_labeled.csv').replace('.parquet', '_labeled.parquet')
                    if format_type == 'csv':
                        df.to_csv(labeled_path, index=False)
                    else:
                        df.to_parquet(labeled_path, index=False)
                    
                    self.show_output(self.analyze_output, f" Predictions complete! Saved to {labeled_path}\n", 'success')
                    dataset_path = labeled_path
                else:
                    self.show_output(self.analyze_output, f"Dataset already labeled, skipping prediction step.\n", 'accent')
                
                self.show_output(self.analyze_output, f"Step 3: Building reports for {ticker}...\n", 'accent')
                
                # Build reports
                output_dir = Path("reports")
                
                reports = build_reports(
                    labeled_path=dataset_path,
                    ticker=ticker,
                    out_dir=str(output_dir),
                    window=window,
                    file_format=format_type
                )
                
                # Load results
                daily_file = reports['daily']
                alerts_file = reports['alerts']
                
                if format_type == 'csv':
                    daily_df = pd.read_csv(daily_file)
                    alerts_df = pd.read_csv(alerts_file)
                else:
                    daily_df = pd.read_parquet(daily_file)
                    alerts_df = pd.read_parquet(alerts_file)
                
                # Check if we have data
                if daily_df.empty or daily_df['article_count'].sum() == 0:
                    self.show_output(self.analyze_output, f" No articles found for ticker {ticker}. Make sure the ticker matches the fetched data.\n", 'warning')
                    self.set_status(f" No articles for {ticker}", 'warning')
                    return
                
                # Calculate detailed statistics
                total_articles = int(daily_df['article_count'].sum())
                avg_sentiment = daily_df['mean_score'].mean()
                sentiment_label = " Positive" if avg_sentiment > 0.2 else " Negative" if avg_sentiment < -0.2 else " Neutral"
                
                # Get sentiment distribution from original labeled data
                if dataset_path.endswith('.csv'):
                    labeled_df = pd.read_csv(dataset_path)
                else:
                    labeled_df = pd.read_parquet(dataset_path)
                
                sentiment_counts = labeled_df['weak_label'].value_counts()
                pct_positive = (sentiment_counts.get('positive', 0) / len(labeled_df) * 100) if len(labeled_df) > 0 else 0
                pct_neutral = (sentiment_counts.get('neutral', 0) / len(labeled_df) * 100) if len(labeled_df) > 0 else 0
                pct_negative = (sentiment_counts.get('negative', 0) / len(labeled_df) * 100) if len(labeled_df) > 0 else 0
                
                # Format daily metrics with better display
                daily_display = daily_df[['date', 'mean_score', 'article_count', 'pct_pos', 'pct_neg']].copy()
                daily_display['mean_score'] = daily_display['mean_score'].round(3)
                daily_display['pct_pos'] = (daily_display['pct_pos'] * 100).round(1)
                daily_display['pct_neg'] = (daily_display['pct_neg'] * 100).round(1)
                daily_display.columns = ['Date', 'Sentiment', 'Articles', 'Positive%', 'Negative%']
                
                result = f"""
 SENTIMENT ANALYSIS RESULTS


OVERALL SUMMARY

Ticker: {ticker.upper()}
Total Articles Analyzed: {total_articles}
Analysis Period: {len(daily_df)} days
Date Range: {daily_df['date'].min()} to {daily_df['date'].max()}

Average Sentiment Score: {avg_sentiment:.3f} {sentiment_label}
Rolling Window: {window} days

SENTIMENT DISTRIBUTION:
   Positive: {pct_positive:.1f}% ({sentiment_counts.get('positive', 0)} articles)
   Neutral:  {pct_neutral:.1f}% ({sentiment_counts.get('neutral', 0)} articles)
   Negative: {pct_negative:.1f}% ({sentiment_counts.get('negative', 0)} articles)


DAILY BREAKDOWN (All Days)

{daily_display.to_string(index=False)}


ALERTS & ANOMALIES

Total Alerts Detected: {len(alerts_df)}
{alerts_df.to_string(index=False) if len(alerts_df) > 0 else " No significant sentiment spikes detected."}


 REPORTS SAVED TO:

¢ Daily Metrics: {daily_file}
¢ Alerts Report: {alerts_file}
¢ Labeled Dataset: {dataset_path}
                """
                
                self.show_output(self.analyze_output, result, 'success')
                self.set_status(f" Analyzed {total_articles} articles - {len(alerts_df)} alerts", 'success')
                
            except Exception as e:
                self.show_output(self.analyze_output, f" Error: {str(e)}\n", 'error')
                self.set_status(" Analysis failed", 'error')
            finally:
                self.is_processing = False
        
        thread = threading.Thread(target=task, daemon=True)
        thread.start()
    
    def predict_sentiment(self):
        """Predict sentiment for text"""
        if self.is_processing:
            messagebox.showwarning("Processing", "Already processing. Please wait...")
            return
        
        def task():
            self.is_processing = True
            self.set_status(" Predicting sentiment...", 'accent')
            
            try:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                import torch
                
                model_path = "models/Colab/finetuned_improved"
                
                if not Path(model_path).exists():
                    self.show_output(self.test_output, " Error: Model not found at models/Colab/finetuned_improved/\n", 'error')
                    return
                
                text = self.test_input.get(1.0, tk.END).strip()
                if not text:
                    self.show_output(self.test_output, " Error: Please enter some text\n", 'error')
                    return
                
                self.show_output(self.test_output, "Loading model...\n", 'accent')
                
                # Load model
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForSequenceClassification.from_pretrained(model_path)
                model.eval()
                
                # Predict
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1)[0]
                    prediction = outputs.logits.argmax(dim=-1).item()
                
                labels = ["negative", "neutral", "positive"]
                predicted_label = labels[prediction]
                confidence = probs[prediction].item()
                
                result = f"""
 PREDICTION: {predicted_label.upper()} ({confidence:.2%} confidence)

PROBABILITIES:
   Negative: {probs[0]:.2%}
   Neutral:  {probs[1]:.2%}
   Positive: {probs[2]:.2%}

INPUT TEXT:
{text}
                """
                
                self.show_output(self.test_output, result, 'success')
                self.set_status(f" Prediction: {predicted_label.upper()}", 'success')
                
            except Exception as e:
                self.show_output(self.test_output, f" Error: {str(e)}\n", 'error')
                self.set_status(" Prediction failed", 'error')
            finally:
                self.is_processing = False
        
        thread = threading.Thread(target=task, daemon=True)
        thread.start()
    
    def show_output(self, widget, text, msg_type='text'):
        """Show output in text widget"""
        widget.delete(1.0, tk.END)
        widget.insert(1.0, text)
        
        # Optional: Color-code based on type
        # widget.tag_config('success', foreground=self.colors['success'])
        # widget.tag_add('success', 1.0, tk.END) if msg_type == 'success' else None


def main():
    root = tk.Tk()
    app = SentimentAnalysisApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
