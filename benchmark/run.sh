#!/bin/bash
# Genesis Benchmark Suite
# Run comprehensive benchmarks comparing Genesis vs PyTorch and generate documentation

echo "🚀 Genesis Benchmark Suite"
echo "=========================="

# Set GPU device (change this to your available GPU)
export CUDA_VISIBLE_DEVICES=0

# Check if CUDA is available
if ! python -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>/dev/null; then
    echo "❌ CUDA not available or PyTorch not installed"
    exit 1
fi

# Create timestamp for this benchmark run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
echo "🕒 Benchmark run: $TIMESTAMP"

# Create docs/benchmark directory if it doesn't exist
DOCS_BENCHMARK_DIR="../docs/benchmark"
mkdir -p "$DOCS_BENCHMARK_DIR"

echo "📊 Running comprehensive benchmarks and generating documentation..."
echo ""

# Core operations benchmark - comprehensive mode
echo "1. Element-wise Operations Benchmark (Comprehensive)"
echo "---------------------------------------------------"
python bench_ops.py --timing both
echo ""

# Core operations benchmark - specific categories for detailed analysis
echo "2. Element-wise Operations by Category"
echo "-------------------------------------"
for category in "element" "activation" "reduction" "matrix"; do
    echo "   Testing category: $category"
    python bench_ops.py --category "$category" --fast
done
echo ""

# Qwen model benchmarks
echo "3. Qwen Model Benchmarks"
echo "-----------------------"
if python -c "from genesis.models.qwen import QwenForCausalLM; print('Qwen model available')" 2>/dev/null; then
    echo "   Testing Qwen 0.5B model..."
    python bench_qwen.py --size 0.5B --batch-size 1,2,4 --seq-len 128,256,512 --fast
    echo ""
else
    echo "   ⚠️  Qwen model not available, skipping..."
    echo ""
fi

# Generate index file for benchmark results
echo "4. Generating Benchmark Index"
echo "----------------------------"
python << 'EOF'
import os
import glob
from datetime import datetime

docs_dir = "../docs/benchmark"
if not os.path.exists(docs_dir):
    os.makedirs(docs_dir)

# Get all benchmark files
benchmark_files = []
for pattern in ["operations_*.md", "qwen_model_*.md"]:
    benchmark_files.extend(glob.glob(os.path.join(docs_dir, pattern)))

benchmark_files.sort(key=os.path.getmtime, reverse=True)  # Sort by modification time, newest first

# Generate index.md
index_path = os.path.join(docs_dir, "index.md")
with open(index_path, 'w') as f:
    f.write("# Genesis Benchmark Reports\n\n")
    f.write(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write("This directory contains comprehensive performance benchmark reports comparing Genesis with PyTorch.\n\n")
    
    # Operations benchmarks
    f.write("## Operations Benchmarks\n\n")
    f.write("Detailed performance analysis of individual operations and operation categories.\n\n")
    
    ops_files = [f for f in benchmark_files if "operations_" in os.path.basename(f)]
    if ops_files:
        f.write("| Report | Type |\n")
        f.write("|--------|------|\n")
        for file_path in ops_files:
            filename = os.path.basename(file_path)
            # Determine report type from filename
            if "comprehensive" in filename:
                report_type = "Comprehensive"
            elif any(cat in filename for cat in ["element", "activation", "reduction", "matrix"]):
                cat_name = next((cat for cat in ["element", "activation", "reduction", "matrix"] if cat in filename), "category")
                report_type = f"{cat_name.title()} Category"
            else:
                report_type = "Operations"
            
            f.write(f"| [{filename}](./{filename}) | {report_type} |\n")
    else:
        f.write("No operations benchmark reports available.\n")
    
    # Qwen model benchmarks
    f.write("\n## Qwen Model Benchmarks\n\n")
    f.write("End-to-end performance analysis of Qwen language models.\n\n")
    
    qwen_files = [f for f in benchmark_files if "qwen_model_" in os.path.basename(f)]
    if qwen_files:
        f.write("| Report | Model Size |\n")
        f.write("|--------|------------|\n")
        for file_path in qwen_files:
            filename = os.path.basename(file_path)
            # Extract model size from filename
            try:
                model_part = filename.replace("qwen_model_", "").replace(".md", "")
                model_size = model_part.replace("p", ".")
                f.write(f"| [{filename}](./{filename}) | {model_size}B |\n")
            except:
                f.write(f"| [{filename}](./{filename}) | Unknown |\n")
    else:
        f.write("No Qwen model benchmark reports available.\n")
    
    f.write("\n## About These Benchmarks\n\n")
    f.write("### Operations Benchmarks\n")
    f.write("- **Comprehensive**: Tests all available operations across multiple tensor sizes\n")
    f.write("- **Category-specific**: Focused analysis on specific operation types\n")
    f.write("- **Metrics**: Speedup vs PyTorch, memory bandwidth utilization, reliability scores\n\n")
    
    f.write("### Qwen Model Benchmarks\n")
    f.write("- **End-to-end**: Complete model forward and backward pass timing\n")
    f.write("- **Memory analysis**: Peak memory usage comparison\n")
    f.write("- **Scalability**: Performance across different batch sizes and sequence lengths\n\n")
    
    f.write("### Performance Indicators\n")
    f.write("- 🟢 **Excellent (≥90%)**: Genesis performs at 90% or better vs PyTorch\n")
    f.write("- 🟡 **Good (70-90%)**: Acceptable performance gap\n")
    f.write("- 🟠 **Fair (50-70%)**: Notable performance gap, optimization recommended\n")
    f.write("- 🔴 **Poor (20-50%)**: Significant optimization needed\n")
    f.write("- ❌ **Critical (<20%)**: Major performance issues\n\n")
    
    f.write("### How to Run Benchmarks\n\n")
    f.write("```bash\n")
    f.write("# Run all benchmarks\n")
    f.write("cd benchmark\n")
    f.write("./run.sh\n\n")
    f.write("# Run specific operation category\n")
    f.write("python bench_ops.py --category element\n\n")
    f.write("# Run Qwen model benchmark\n")
    f.write("python bench_qwen.py --size 0.5B --batch-size 1,2,4\n")
    f.write("```\n")

print(f"📄 Generated benchmark index: {index_path}")
EOF

echo "✅ All benchmarks completed!"
echo ""
echo "📁 Documentation generated in: $DOCS_BENCHMARK_DIR"
echo "📊 Genesis benchmark reports are ready for viewing"
echo "🔗 Check docs/benchmark/index.md for a complete overview"
echo ""
echo "📈 Performance summary:"
echo "   - Operations benchmarks test individual operations and categories"
echo "   - Qwen model benchmarks test end-to-end model performance"
echo "   - All reports include detailed analysis and optimization recommendations"