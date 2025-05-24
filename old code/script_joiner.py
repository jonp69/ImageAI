
#!/usr/bin/env python3
"""
Python Script Joiner - Combines multiple Python files into a single script
"""

import os
import sys
import argparse
import ast
from pathlib import Path
from typing import List, Set, Dict, Optional

class PythonScriptJoiner:
    def __init__(self):
        self.imports = set()
        self.from_imports = {}
        self.functions = []
        self.classes = []
        self.variables = []
        self.main_blocks = []
        self.docstrings = []
        
    def extract_components(self, file_path: str, content: str) -> Dict:
        """Extract different components from a Python file"""
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            print(f"Warning: Could not parse {file_path}: {e}")
            return {"raw_content": content}
        
        components = {
            "imports": [],
            "from_imports": [],
            "functions": [],
            "classes": [],
            "variables": [],
            "other": [],
            "main_block": None
        }
        
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    components["imports"].append(f"import {alias.name}")
                    
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                names = [alias.name for alias in node.names]
                components["from_imports"].append(f"from {module} import {', '.join(names)}")
                
            elif isinstance(node, ast.FunctionDef):
                components["functions"].append(ast.get_source_segment(content, node))
                
            elif isinstance(node, ast.ClassDef):
                components["classes"].append(ast.get_source_segment(content, node))
                
            elif isinstance(node, ast.Assign):
                components["variables"].append(ast.get_source_segment(content, node))
                
            elif isinstance(node, ast.If) and isinstance(node.test, ast.Compare):
                # Check for if __name__ == "__main__":
                if (isinstance(node.test.left, ast.Name) and 
                    node.test.left.id == "__name__" and
                    any(isinstance(comp, ast.Constant) and comp.value == "__main__" 
                        for comp in node.test.comparators)):
                    components["main_block"] = ast.get_source_segment(content, node)
                else:
                    components["other"].append(ast.get_source_segment(content, node))
            else:
                segment = ast.get_source_segment(content, node)
                if segment:
                    components["other"].append(segment)
        
        return components
    
    def get_files_to_process(self, folder_path: str = ".", 
                           exclude_self: bool = True,
                           exclude_files: List[str] = None,
                           include_pattern: str = "*.py") -> List[str]:
        """Get list of Python files to process, optionally excluding this script"""
        
        if exclude_files is None:
            exclude_files = []
        
        # Get the current script's filename
        current_script = os.path.basename(__file__) if exclude_self else None
        
        # Find all Python files
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"Folder {folder_path} does not exist")
        
        files = list(folder.glob(include_pattern))
        
        # Filter files
        filtered_files = []
        for file_path in files:
            file_name = file_path.name
            
            # Skip if it's the current script
            if exclude_self and file_name == current_script:
                print(f"Excluding self: {file_name}")
                continue
                
            # Skip if it's in exclude list
            if file_name in exclude_files:
                print(f"Excluding: {file_name}")
                continue
                
            # Skip if it's the output file (if it exists)
            if file_name.endswith("_combined.py") or file_name.endswith("_joined.py"):
                print(f"Excluding output file: {file_name}")
                continue
                
            filtered_files.append(str(file_path))
        
        return sorted(filtered_files)
    
    def join_scripts(self, file_paths: List[str], output_path: str, 
                    include_headers: bool = True, 
                    merge_main_blocks: bool = True,
                    sort_imports: bool = True) -> None:
        """Join multiple Python scripts into one"""
        
        all_imports = set()
        all_from_imports = set()
        all_functions = []
        all_classes = []
        all_variables = []
        all_other = []
        all_main_blocks = []
        
        output_lines = []
        
        # Add header comment
        if include_headers:
            output_lines.extend([
                "#!/usr/bin/env python3",
                '"""',
                "Combined Python Script",
                f"Generated from: {', '.join([os.path.basename(f) for f in file_paths])}",
                f"Total files combined: {len(file_paths)}",
                '"""',
                ""
            ])
        
        # Process each file
        for file_path in file_paths:
            if not os.path.exists(file_path):
                print(f"Warning: File {file_path} does not exist, skipping...")
                continue
                
            print(f"Processing: {os.path.basename(file_path)}")
                
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if include_headers:
                output_lines.extend([
                    f"# " + "="*50,
                    f"# Content from: {os.path.basename(file_path)}",
                    f"# " + "="*50,
                    ""
                ])
            
            components = self.extract_components(file_path, content)
            
            # Handle raw content if parsing failed
            if "raw_content" in components:
                output_lines.extend([
                    "# Raw content (could not parse):",
                    components["raw_content"],
                    ""
                ])
                continue
            
            # Collect components
            all_imports.update(components["imports"])
            all_from_imports.update(components["from_imports"])
            all_functions.extend(components["functions"])
            all_classes.extend(components["classes"])
            all_variables.extend(components["variables"])
            all_other.extend(components["other"])
            
            if components["main_block"]:
                all_main_blocks.append((file_path, components["main_block"]))
        
        # Write imports
        if all_imports or all_from_imports:
            output_lines.extend([
                "# Imports",
                "# " + "-"*30,
                ""
            ])
            if sort_imports:
                for imp in sorted(all_imports):
                    output_lines.append(imp)
                for imp in sorted(all_from_imports):
                    output_lines.append(imp)
            else:
                output_lines.extend(all_imports)
                output_lines.extend(all_from_imports)
            output_lines.append("")
        
        # Write variables
        if all_variables:
            output_lines.extend([
                "# Global Variables",
                "# " + "-"*30,
                ""
            ])
            for var in all_variables:
                output_lines.append(var)
            output_lines.append("")
        
        # Write classes
        if all_classes:
            output_lines.extend([
                "# Classes",
                "# " + "-"*30,
                ""
            ])
            for cls in all_classes:
                output_lines.extend([cls, ""])
        
        # Write functions
        if all_functions:
            output_lines.extend([
                "# Functions",
                "# " + "-"*30,
                ""
            ])
            for func in all_functions:
                output_lines.extend([func, ""])
        
        # Write other code
        if all_other:
            output_lines.extend([
                "# Other Code",
                "# " + "-"*30,
                ""
            ])
            for other in all_other:
                output_lines.extend([other, ""])
        
        # Handle main blocks
        if all_main_blocks:
            if merge_main_blocks:
                output_lines.extend([
                    "# Main Execution",
                    "# " + "-"*30,
                    "",
                    'if __name__ == "__main__":',
                    "    # Combined main blocks from all files",
                    ""
                ])
                
                for file_path, main_block in all_main_blocks:
                    output_lines.extend([
                        f"    # From {os.path.basename(file_path)}:",
                        ""
                    ])
                    # Extract content from main block and indent
                    main_lines = main_block.split('\n')[1:]  # Skip the if line
                    for line in main_lines:
                        if line.strip():
                            # Add extra indentation
                            output_lines.append("    " + line)
                        else:
                            output_lines.append("")
                    output_lines.append("")
            else:
                output_lines.extend([
                    "# Main Blocks (kept separate)",
                    "# " + "-"*30,
                    ""
                ])
                for file_path, main_block in all_main_blocks:
                    output_lines.extend([
                        f"# Main block from {os.path.basename(file_path)}:",
                        main_block,
                        ""
                    ])
        
        # Write to output file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_lines))
        
        print(f"\n✅ Successfully joined {len(file_paths)} files into {output_path}")
        print(f"📊 Total lines: {len(output_lines)}")
        print(f"📁 Output size: {os.path.getsize(output_path)} bytes")

def main():
    parser = argparse.ArgumentParser(description="Join multiple Python scripts into one")
    parser.add_argument("files", nargs="*", help="Python files to join (optional if using --folder)")
    parser.add_argument("-o", "--output", help="Output file path (default: combined.py)")
    parser.add_argument("-f", "--folder", help="Process all .py files in this folder (default: current directory)")
    parser.add_argument("--include-self", action="store_true", help="Include this script in the combination")
    parser.add_argument("--exclude", nargs="+", help="Files to exclude by name")
    parser.add_argument("--no-headers", action="store_true", help="Don't include file headers")
    parser.add_argument("--separate-main", action="store_true", help="Keep main blocks separate")
    parser.add_argument("--no-sort-imports", action="store_true", help="Don't sort imports")
    parser.add_argument("--pattern", default="*.py", help="File pattern to match (default: *.py)")
    
    args = parser.parse_args()
    
    # Set default output if not provided
    if not args.output:
        args.output = "combined.py"
    
    # Determine files to process
    files = []
    
    if args.folder or not args.files:
        # Process folder mode
        folder = args.folder or "."
        exclude_files = args.exclude or []
        
        joiner = PythonScriptJoiner()
        files = joiner.get_files_to_process(
            folder_path=folder,
            exclude_self=not args.include_self,
            exclude_files=exclude_files,
            include_pattern=args.pattern
        )
        
        if not files:
            print("❌ No Python files found to process")
            return
            
        print(f"📂 Found {len(files)} files to process:")
        for f in files:
            print(f"   • {os.path.basename(f)}")
        print()
        
    else:
        # Use specified files
        files = args.files
    
    if not files:
        print("❌ No files to process")
        return
    
    # Check if output file would overwrite an input file
    abs_output = os.path.abspath(args.output)
    for file_path in files:
        if os.path.abspath(file_path) == abs_output:
            print(f"❌ Error: Output file {args.output} would overwrite input file {file_path}")
            return
    
    joiner = PythonScriptJoiner()
    joiner.join_scripts(
        files, 
        args.output,
        include_headers=not args.no_headers,
        merge_main_blocks=not args.separate_main,
        sort_imports=not args.no_sort_imports
    )

if __name__ == "__main__":
    main()

