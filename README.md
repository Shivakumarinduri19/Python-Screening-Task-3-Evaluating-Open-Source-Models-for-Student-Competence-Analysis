# Python Screening Task 3: Evaluating Open Source Models for Student Competence Analysis

## 📌 Task Overview

This repository contains my submission for **Python Screening Task 3** assigned by the FOSSEE Python team. The task focuses on exploring open-source AI models that can be adapted to generate meaningful prompts or insights for **high-level student competence analysis in Python learning**.

## 📝 Research Plan

My approach was to:

1. **Survey open-source models** such as CodeLlama-Python, StarCoder, and SantaCoder that are trained for code understanding and reasoning.
2. **Evaluate their ability** to:

   * Analyze student-written Python code
   * Generate conceptual prompts without revealing direct solutions
   * Identify gaps in reasoning or misconceptions
   * Encourage deeper learning
3. **Benchmark results** with student-level Python problems to assess accuracy, interpretability, and cost-effectiveness.

**Key Findings:**

* **CodeLlama-Python** is the most promising due to its strong optimization for code analysis and community support.
* **Limitations**: Can be verbose or overly technical, requires significant compute resources for larger models.
* **Trade-offs**: Larger models = better accuracy but higher cost; smaller models = accessible but less consistent.

## 💡 Reasoning

* **What makes a model suitable?**
  A good model must balance accuracy (detecting real issues), interpretability (student-friendly explanations), and efficiency (low cost for classroom settings).

* **How to test prompt quality?**
  By comparing model-generated prompts against educator-designed rubrics and checking if they encourage reasoning instead of spoon-feeding answers.

* **Key trade-offs:**
  Accuracy vs. interpretability vs. cost. For example, CodeLlama-34B provides better results but is expensive to run, while CodeLlama-7B is cheaper but less reliable.

* **Why CodeLlama?**
  It is optimized for programming, well-supported, and provides a strong foundation for adaptation in education.

## ⚙️ Setup Instructions

1. Clone this repository:

   ```bash
   git clone [your_repo_link]
   cd [repo_name]
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Run the evaluation script (if included):

   ```bash
   python evaluate_model.py
   ```

## 🔗 References

* [CodeLlama (Meta AI)](https://huggingface.co/codellama)
* [StarCoder (Hugging Face)](https://huggingface.co/bigcode/starcoder)
* [SantaCoder](https://huggingface.co/bigcode/santacoder)

## 🙋 Author

SHIVA KUMAR INDURI
Note: I used AI tools to enhance the clarity and structure of this submission, but the reasoning and evaluation are my own.


