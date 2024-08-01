from django.shortcuts import render
from django.http import JsonResponse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sqlparse

model_name = "defog/llama-3-sqlcoder-8b"
tokenizer = AutoTokenizer.from_pretrained(model_name)

torch.cuda.is_available()
available_memory = torch.cuda.get_device_properties(0).total_memory

if available_memory > 20e9:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        use_cache=True,
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        load_in_4bit=True,
        device_map="auto",
        use_cache=True,
    )

prompt = """user

Generate a SQL query to answer this question: `{question}`

DDL statements:

CREATE TABLE products (
  product_id INTEGER PRIMARY KEY, -- Unique ID for each product
  name VARCHAR(50), -- Name of the product
  price DECIMAL(10,2), -- Price of each unit of the product
  quantity INTEGER  -- Current quantity in stock
);

CREATE TABLE customers (
   customer_id INTEGER PRIMARY KEY, -- Unique ID for each customer
   name VARCHAR(50), -- Name of the customer
   address VARCHAR(100) -- Mailing address of the customer
);

CREATE TABLE salespeople (
  salesperson_id INTEGER PRIMARY KEY, -- Unique ID for each salesperson
  name VARCHAR(50), -- Name of the salesperson
  region VARCHAR(50) -- Geographic sales region
);

CREATE TABLE sales (
  sale_id INTEGER PRIMARY KEY, -- Unique ID for each sale
  product_id INTEGER, -- ID of product sold
  customer_id INTEGER,  -- ID of customer who made purchase
  salesperson_id INTEGER, -- ID of salesperson who made the sale
  sale_date DATE, -- Date the sale occurred
  quantity INTEGER -- Quantity of product sold
);

CREATE TABLE product_suppliers (
  supplier_id INTEGER PRIMARY KEY, -- Unique ID for each supplier
  product_id INTEGER, -- Product ID supplied
  supply_price DECIMAL(10,2) -- Unit price charged by supplier
);

-- sales.product_id can be joined with products.product_id
-- sales.customer_id can be joined with customers.customer_id
-- sales.salesperson_id can be joined with salespeople.salesperson_id
-- product_suppliers.product_id can be joined with products.product_id

The following SQL query best answers the question `{question}`:
```sql
"""

def generate_query(question):
    updated_prompt = prompt.format(question=question)
    inputs = tokenizer(updated_prompt, return_tensors="pt").to("cuda")
    generated_ids = model.generate(
        **inputs,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=400,
        do_sample=False,
        num_beams=1,
        temperature=0.0,
        top_p=1,
    )
    outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    return outputs[0].split("```sql")[1].split(";")[0]

def index(request):
    if request.method == 'POST':
        question = request.POST.get('question')
        generated_sql = generate_query(question)
        formatted_sql = sqlparse.format(generated_sql, reindent=True)
        return JsonResponse({'sql_query': formatted_sql})

    return render(request, 'index.html')
