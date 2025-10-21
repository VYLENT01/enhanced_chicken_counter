#!/usr/bin/env python3
"""
Enhanced Chicken Counter - Interactive Launcher
Interactive menu system for configuring and running the chicken counter

Features:
- Source selection (webcam, video file)
- Model selection and configuration
- Performance settings
- Real-time configuration updates
"""

import os
import sys
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.prompt import Prompt, Confirm, IntPrompt, FloatPrompt
    from rich.columns import Columns
    from rich.text import Text
    from rich.layout import Layout
    from dotenv import load_dotenv, set_key
    
    # Import our modules
    from main_counter import ChickenCounterSystem
    from data_logger import DataLogger
    from ai_analyzer import AIAnalyzer
    
except ImportError as e:
    print(f"❌ Missing required dependencies: {e}")
    print("📦 Please run: python install.py")
    sys.exit(1)

class InteractiveLauncher:
    """Interactive launcher for Enhanced Chicken Counter"""
    
    def __init__(self):
        self.console = Console()
        self.env_file = Path(".env")
        self.project_root = Path.cwd()
        
        # Load current environment
        if self.env_file.exists():
            load_dotenv(self.env_file)
        
        # Available models
        self.available_models = {
            "yolov8n.pt": {"name": "YOLOv8 Nano", "size": "6MB", "speed": "Fastest", "accuracy": "Good"},
            "yolov8s.pt": {"name": "YOLOv8 Small", "size": "22MB", "speed": "Fast", "accuracy": "Better"}, 
            "yolov8m.pt": {"name": "YOLOv8 Medium", "size": "50MB", "speed": "Medium", "accuracy": "Best"},
            "yolov8l.pt": {"name": "YOLOv8 Large", "size": "88MB", "speed": "Slow", "accuracy": "Excellent"},
            "yolov8x.pt": {"name": "YOLOv8 XLarge", "size": "136MB", "speed": "Slowest", "accuracy": "Maximum"}
        }
        
        # Configuration
        self.config = self._load_current_config()
    
    def _load_current_config(self) -> Dict[str, str]:
        """Load current configuration from environment"""
        return {
            'INPUT_SOURCE': os.getenv('INPUT_SOURCE', '0'),
            'DEFAULT_MODEL': os.getenv('DEFAULT_MODEL', 'yolov8n.pt'),
            'MODEL_CONFIDENCE': os.getenv('MODEL_CONFIDENCE', '0.25'),
            'USE_GPU': os.getenv('USE_GPU', 'true'),
            'SHOW_VIDEO': os.getenv('SHOW_VIDEO', 'true'),
            'WINDOW_WIDTH': os.getenv('WINDOW_WIDTH', '1280'),
            'WINDOW_HEIGHT': os.getenv('WINDOW_HEIGHT', '720'),
            'ENABLE_AI_ANALYSIS': os.getenv('ENABLE_AI_ANALYSIS', 'true'),
            'AI_MODEL': os.getenv('AI_MODEL', 'anthropic/claude-3-haiku'),
            'OPENROUTER_API_KEY': os.getenv('OPENROUTER_API_KEY', '')
        }
    
    def show_welcome(self):
        """Display welcome screen"""
        welcome_text = """
[bold green]🐔 Enhanced Chicken Counter System[/bold green]
[cyan]Interactive Configuration & Launch Menu[/cyan]

[dim]Version 1.0.0 | Powered by YOLOv8 + ByteTrack + AI[/dim]
        """
        
        panel = Panel(
            welcome_text,
            title="Welcome",
            border_style="green",
            padding=(1, 2)
        )
        
        self.console.print(panel)
    
    def show_current_config(self):
        """Display current configuration"""
        table = Table(title="Current Configuration", border_style="blue")
        table.add_column("Setting", style="cyan", width=20)
        table.add_column("Value", style="yellow", width=30)
        table.add_column("Description", style="dim", width=40)
        
        config_descriptions = {
            'INPUT_SOURCE': 'Video input source',
            'DEFAULT_MODEL': 'AI detection model',
            'MODEL_CONFIDENCE': 'Detection confidence threshold',
            'USE_GPU': 'GPU acceleration',
            'SHOW_VIDEO': 'Show real-time video',
            'WINDOW_WIDTH': 'Display window width',
            'WINDOW_HEIGHT': 'Display window height',
            'ENABLE_AI_ANALYSIS': 'AI behavior analysis',
            'AI_MODEL': 'AI analysis model'
        }
        
        for key, value in self.config.items():
            if key == 'OPENROUTER_API_KEY':
                display_value = "***configured***" if value else "not set"
            else:
                display_value = value
                
            description = config_descriptions.get(key, '')
            table.add_row(key, display_value, description)
        
        self.console.print(table)
    
    def select_input_source(self):
        """Select video input source"""
        self.console.print("\n[bold cyan]📹 Select Input Source[/bold cyan]")
        
        options = [
            "Webcam (Real-time)",
            "Video File", 
            "Custom Source (IP Camera, etc.)"
        ]
        
        for i, option in enumerate(options, 1):
            self.console.print(f"  {i}. {option}")
        
        choice = IntPrompt.ask("Choose input source", choices=["1", "2", "3"], default=1)
        
        if choice == 1:
            # Webcam
            camera_id = IntPrompt.ask("Enter camera ID", default=0)
            self.config['INPUT_SOURCE'] = str(camera_id)
            self.console.print(f"✅ Selected: Webcam (ID: {camera_id})")
            
        elif choice == 2:
            # Video file
            self.console.print("\n[yellow]📁 Video File Selection[/yellow]")
            
            # Show available video files
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
            video_files = []
            
            for ext in video_extensions:
                video_files.extend(self.project_root.glob(f"*{ext}"))
                video_files.extend(self.project_root.glob(f"**/*{ext}"))
            
            if video_files:
                self.console.print("\n[dim]Found video files:[/dim]")
                for i, video_file in enumerate(video_files[:10], 1):  # Show max 10
                    self.console.print(f"  {i}. {video_file.name} ({video_file.parent})")
                
                self.console.print(f"  {len(video_files)+1}. Enter custom path")
                
                if len(video_files) > 10:
                    self.console.print(f"  ... and {len(video_files)-10} more files")
                
                max_choice = min(len(video_files), 10) + 1
                file_choice = IntPrompt.ask(
                    f"Choose video file (1-{max_choice})",
                    choices=[str(i) for i in range(1, max_choice+1)]
                )
                
                if file_choice <= len(video_files):
                    selected_file = video_files[file_choice-1]
                    self.config['INPUT_SOURCE'] = str(selected_file)
                    self.console.print(f"✅ Selected: {selected_file.name}")
                else:
                    custom_path = Prompt.ask("Enter video file path")
                    if Path(custom_path).exists():
                        self.config['INPUT_SOURCE'] = custom_path
                        self.console.print(f"✅ Selected: {custom_path}")
                    else:
                        self.console.print(f"❌ File not found: {custom_path}")
                        return self.select_input_source()
            else:
                custom_path = Prompt.ask("Enter video file path")
                if Path(custom_path).exists():
                    self.config['INPUT_SOURCE'] = custom_path
                    self.console.print(f"✅ Selected: {custom_path}")
                else:
                    self.console.print(f"❌ File not found: {custom_path}")
                    return self.select_input_source()
        
        elif choice == 3:
            # Custom source
            custom_source = Prompt.ask("Enter custom source (IP camera URL, etc.)")
            self.config['INPUT_SOURCE'] = custom_source
            self.console.print(f"✅ Selected: {custom_source}")
    
    def select_model(self):
        """Select AI model"""
        self.console.print("\n[bold cyan]🎯 Select Detection Model[/bold cyan]")
        
        table = Table(border_style="cyan")
        table.add_column("Option", width=8)
        table.add_column("Model", width=15)
        table.add_column("Size", width=8)
        table.add_column("Speed", width=10)
        table.add_column("Accuracy", width=12)
        table.add_column("Use Case", width=25)
        
        use_cases = {
            "yolov8n.pt": "Real-time, edge devices",
            "yolov8s.pt": "Balanced applications", 
            "yolov8m.pt": "High accuracy needs",
            "yolov8l.pt": "Research applications",
            "yolov8x.pt": "Critical precision"
        }
        
        models = list(self.available_models.keys())
        for i, model_key in enumerate(models, 1):
            model_info = self.available_models[model_key]
            use_case = use_cases.get(model_key, "General purpose")
            
            table.add_row(
                str(i),
                model_info["name"],
                model_info["size"],
                model_info["speed"],
                model_info["accuracy"],
                use_case
            )
        
        self.console.print(table)
        
        choice = IntPrompt.ask(
            f"Choose model (1-{len(models)})",
            choices=[str(i) for i in range(1, len(models)+1)],
            default=1
        )
        
        selected_model = models[choice-1]
        self.config['DEFAULT_MODEL'] = selected_model
        
        model_info = self.available_models[selected_model]
        self.console.print(f"✅ Selected: {model_info['name']} ({model_info['size']})")
    
    def configure_performance(self):
        """Configure performance settings"""
        self.console.print("\n[bold cyan]⚡ Performance Configuration[/bold cyan]")
        
        # GPU usage
        use_gpu = Confirm.ask("Enable GPU acceleration?", default=self.config['USE_GPU'].lower() == 'true')
        self.config['USE_GPU'] = str(use_gpu).lower()
        
        # Confidence threshold
        current_conf = float(self.config['MODEL_CONFIDENCE'])
        confidence = FloatPrompt.ask(
            f"Detection confidence threshold (0.1-0.9)",
            default=current_conf,
            show_default=True
        )
        confidence = max(0.1, min(0.9, confidence))
        self.config['MODEL_CONFIDENCE'] = str(confidence)
        
        # Display settings
        show_video = Confirm.ask("Show real-time video window?", default=self.config['SHOW_VIDEO'].lower() == 'true')
        self.config['SHOW_VIDEO'] = str(show_video).lower()
        
        if show_video:
            # Window size presets
            size_presets = {
                1: ("640", "480", "Small (640x480)"),
                2: ("1280", "720", "HD (1280x720)"),
                3: ("1920", "1080", "Full HD (1920x1080)"),
                4: ("custom", "custom", "Custom size")
            }
            
            self.console.print("\nWindow size presets:")
            for key, (w, h, desc) in size_presets.items():
                self.console.print(f"  {key}. {desc}")
            
            size_choice = IntPrompt.ask("Choose window size", choices=["1", "2", "3", "4"], default=2)
            
            if size_choice == 4:
                width = IntPrompt.ask("Window width", default=int(self.config['WINDOW_WIDTH']))
                height = IntPrompt.ask("Window height", default=int(self.config['WINDOW_HEIGHT']))
                self.config['WINDOW_WIDTH'] = str(width)
                self.config['WINDOW_HEIGHT'] = str(height)
            else:
                width, height, _ = size_presets[size_choice]
                self.config['WINDOW_WIDTH'] = width
                self.config['WINDOW_HEIGHT'] = height
        
        self.console.print("✅ Performance settings configured")
    
    def configure_ai_analysis(self):
        """Configure AI analysis settings"""
        self.console.print("\n[bold cyan]🧠 AI Analysis Configuration[/bold cyan]")
        
        # Check if API key exists
        has_api_key = bool(self.config['OPENROUTER_API_KEY'])
        
        if not has_api_key:
            self.console.print("[yellow]⚠️  No OpenRouter API key found[/yellow]")
            setup_ai = Confirm.ask("Setup AI analysis? (requires OpenRouter API key)")
            
            if setup_ai:
                api_key = Prompt.ask("Enter OpenRouter API key", password=True)
                self.config['OPENROUTER_API_KEY'] = api_key
                has_api_key = True
            else:
                self.config['ENABLE_AI_ANALYSIS'] = 'false'
                self.console.print("✅ AI analysis disabled")
                return
        
        if has_api_key:
            enable_ai = Confirm.ask("Enable AI behavior analysis?", default=self.config['ENABLE_AI_ANALYSIS'].lower() == 'true')
            self.config['ENABLE_AI_ANALYSIS'] = str(enable_ai).lower()
            
            if enable_ai:
                # Model selection
                ai_models = {
                    1: ("anthropic/claude-3-haiku", "Claude 3 Haiku (Fast, Cost-effective)"),
                    2: ("anthropic/claude-3-5-sonnet", "Claude 3.5 Sonnet (High quality)"),
                    3: ("openai/gpt-4o", "GPT-4o (Comprehensive insights)"),
                    4: ("google/gemini-pro-vision", "Gemini Pro Vision (Visual understanding)")
                }
                
                self.console.print("\nAvailable AI models:")
                for key, (model_id, desc) in ai_models.items():
                    self.console.print(f"  {key}. {desc}")
                
                ai_choice = IntPrompt.ask("Choose AI model", choices=["1", "2", "3", "4"], default=1)
                selected_ai_model = ai_models[ai_choice][0]
                self.config['AI_MODEL'] = selected_ai_model
                
                self.console.print(f"✅ AI analysis configured with {ai_models[ai_choice][1]}")
        
    def save_configuration(self):
        """Save configuration to .env file"""
        try:
            # Ensure .env file exists
            if not self.env_file.exists():
                self.env_file.touch()
            
            # Update .env file
            for key, value in self.config.items():
                set_key(str(self.env_file), key, value)
            
            self.console.print("✅ Configuration saved to .env file")
            return True
            
        except Exception as e:
            self.console.print(f"❌ Failed to save configuration: {e}")
            return False
    
    def show_launch_summary(self):
        """Show launch summary"""
        self.console.print("\n[bold green]🚀 Launch Summary[/bold green]")
        
        # Create summary table
        table = Table(title="System Configuration", border_style="green")
        table.add_column("Component", style="cyan", width=20)
        table.add_column("Setting", style="yellow", width=40)
        
        # Input source
        source = self.config['INPUT_SOURCE']
        if source.isdigit():
            source_desc = f"Webcam (ID: {source})"
        else:
            source_desc = f"Video file: {Path(source).name}"
        
        table.add_row("Input Source", source_desc)
        
        # Model
        model_key = self.config['DEFAULT_MODEL']
        model_info = self.available_models.get(model_key, {"name": "Custom"})
        table.add_row("Detection Model", f"{model_info['name']} ({model_key})")
        
        # Performance
        gpu_status = "Enabled" if self.config['USE_GPU'].lower() == 'true' else "Disabled"
        table.add_row("GPU Acceleration", gpu_status)
        
        table.add_row("Confidence Threshold", self.config['MODEL_CONFIDENCE'])
        
        # Display
        if self.config['SHOW_VIDEO'].lower() == 'true':
            display_info = f"{self.config['WINDOW_WIDTH']}x{self.config['WINDOW_HEIGHT']}"
        else:
            display_info = "Disabled"
        table.add_row("Video Display", display_info)
        
        # AI Analysis
        ai_status = "Enabled" if self.config['ENABLE_AI_ANALYSIS'].lower() == 'true' else "Disabled"
        if ai_status == "Enabled":
            ai_status += f" ({self.config['AI_MODEL']})"
        table.add_row("AI Analysis", ai_status)
        
        self.console.print(table)
    
    async def launch_system(self):
        """Launch the chicken counter system"""
        try:
            # Re-load environment with new settings
            load_dotenv(self.env_file, override=True)
            
            # Initialize components
            data_logger = DataLogger()
            ai_analyzer = AIAnalyzer()
            counter_system = ChickenCounterSystem(data_logger, ai_analyzer)
            
            # Initialize system
            if not counter_system.initialize():
                self.console.print("❌ Failed to initialize system")
                return False
            
            self.console.print("\n[bold green]✅ System initialized successfully![/bold green]")
            self.console.print("\n[cyan]Starting chicken counting...[/cyan]")
            self.console.print("[dim]Press Ctrl+C to stop[/dim]")
            
            # Start the system
            await counter_system.run()
            
            return True
            
        except KeyboardInterrupt:
            self.console.print("\n🛑 [yellow]Stopped by user[/yellow]")
            return True
        except Exception as e:
            self.console.print(f"\n❌ [red]System error: {e}[/red]")
            return False
    
    def run_interactive_menu(self):
        """Run the main interactive menu"""
        try:
            # Welcome
            self.show_welcome()
            
            # Main menu loop
            while True:
                self.console.print("\n[bold cyan]📋 Main Menu[/bold cyan]")
                
                menu_options = [
                    "1. Show current configuration",
                    "2. Select input source (webcam/video)",
                    "3. Select detection model", 
                    "4. Configure performance settings",
                    "5. Configure AI analysis",
                    "6. Save configuration",
                    "7. Launch chicken counter",
                    "8. Quick launch (use current settings)",
                    "9. Exit"
                ]
                
                for option in menu_options:
                    self.console.print(f"  {option}")
                
                choice = Prompt.ask("Choose an option", choices=[str(i) for i in range(1, 10)], default="1")
                
                if choice == "1":
                    self.show_current_config()
                elif choice == "2":
                    self.select_input_source()
                elif choice == "3":
                    self.select_model()
                elif choice == "4":
                    self.configure_performance()
                elif choice == "5":
                    self.configure_ai_analysis()
                elif choice == "6":
                    self.save_configuration()
                elif choice == "7":
                    if self.save_configuration():
                        self.show_launch_summary()
                        if Confirm.ask("\nLaunch with these settings?"):
                            return asyncio.run(self.launch_system())
                elif choice == "8":
                    self.save_configuration()
                    return asyncio.run(self.launch_system())
                elif choice == "9":
                    self.console.print("👋 Goodbye!")
                    return True
                
        except KeyboardInterrupt:
            self.console.print("\n👋 [yellow]Goodbye![/yellow]")
            return True
        except Exception as e:
            self.console.print(f"❌ [red]Menu error: {e}[/red]")
            return False

def main():
    """Main entry point"""
    launcher = InteractiveLauncher()
    launcher.run_interactive_menu()

if __name__ == "__main__":
    main()