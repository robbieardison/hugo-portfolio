---
title: "Pandas Cheat Sheet"
date: 2022-08-01
tags: ["pandas", "python"]
draft: false
categories:
  - Python
thumbnail: ./images/pandas_cheat_sheet.png
toc: true
---

# Pandas Cheat Sheet: A Comprehensive Guide to Data Manipulation in Python

Pandas, a powerful and versatile library in Python, is the go-to tool for data manipulation and analysis. Whether you're a beginner or an experienced data scientist, this cheat sheet provides a quick reference for common Pandas operations, along with explanations and example codes.

![png](/images/pandas_cheat_sheet.png)

1. **Importing Pandas:**

   - Explanation: Pandas is typically imported with the alias `pd` for brevity.
   - Example Code:
     ```python
     import pandas as pd
     ```

2. **Creating DataFrames:**

   - Explanation: DataFrames are the primary Pandas data structure, representing tabular data.
   - Example Code:
     ```python
     data = {'Name': ['Eric', 'Robby', 'Tina'],
             'Age': [25, 30, 35],
             'City': ['Jakarta', 'Makassar', 'Banjarmasin']}
     df = pd.DataFrame(data)
     ```
   - Output:
     ```
        Name  Age           City
     0  Eric   25       Jakarta
     1    Robby   30  Makassar
     2 Tina   35    Banjarmasin
     ```

3. **Reading Data:**

   - Explanation: Pandas supports various file formats, such as CSV, Excel, and SQL databases.
   - Example Code:

     ```python
     # Reading from CSV
     df_csv = pd.read_csv('data.csv')

     # Reading from Excel
     df_excel = pd.read_excel('data.xlsx')

     # Reading from SQL
     df_sql = pd.read_sql('SELECT * FROM table', connection)
     ```

4. **Exploring Data:**

   - Explanation: Quickly inspect the structure and summary statistics of your data.
   - Example Code:
     ```python
     # Display the first few rows
     print(df.head())
     ```
   - Output:
     ```
        Name  Age           City
     0  Eric   25       Jakarta
     1    Robby   30  Makassar
     2 Tina   35    Banjarmasin
     ```
     ```python
     # Descriptive statistics
     print(df.describe())
     ```
     - Output:
       ```
                  Age
       count   3.000000
       mean   30.000000
       std     5.773503
       min    25.000000
       25%    27.500000
       50%    30.000000
       75%    32.500000
       max    35.000000
       ```
     ```python
     # Data types and missing values
     print(df.info())
     ```
     - Output:
       ```
       <class 'pandas.core.frame.DataFrame'>
       RangeIndex: 3 entries, 0 to 2
       Data columns (total 3 columns):
        #   Column  Non-Null Count  Dtype
       ---  ------  --------------  -----
        0   Name    3 non-null      object
        1   Age     3 non-null      int64
        2   City    3 non-null      object
       dtypes: int64(1), object(2)
       memory usage: 200.0+ bytes
       ```

5. **Indexing and Selecting Data:**

   - Explanation: Access specific rows or columns using labels or numerical indices.
   - Example Code:
     ```python
     # Selecting columns by name
     age_column = df['Age']
     ```
   - Output:
     ```
     0    25
     1    30
     2    35
     Name: Age, dtype: int64
     ```
     ```python
     # Selecting rows by index
     row_1 = df.loc[1]
     ```
     - Output:
       ```
       Name              Robby
       Age                30
       City    Makassar
       Name: 1, dtype: object
       ```
     ```python
     # Filtering data
     young_people = df[df['Age'] < 30]
     ```
     - Output:

   ```
       Name  Age           City
    0  Eric   25       Jakarta
    1    Robby   30  Makassar
   ```

6. **Handling Missing Data:**

   - Explanation: Pandas provides tools for detecting and handling missing values.
   - Example Code:
     ```python
     # Checking for missing values
     print(df.isnull())
     ```
     - Output:
       ```
        Name    Age   City
       0  False  False  False
       1  False  False  False
       2  False  False  False
       ```
     ```python
     # Dropping missing values
     df_clean = df.dropna()
     ```
     - Output: (Same as the original DataFrame if no missing values)
     ```
        Name  Age           City
     0  Eric   25       Jakarta
     1    Robby   30  Makassar
     2 Tina   35    Banjarmasin
     ```
     ```python
     # Filling missing values
     df_filled = df.fillna(0)
     ```
     - Output: (Filled with 0 for missing values)
     ```
        Name  Age           City
     0  Eric   25       Jakarta
     1    Robby   30  Makassar
     2 Tina   35    Banjarmasin
     ```

7. **Grouping and Aggregating:**

   - Explanation: Grouping data by one or more columns and calculating aggregate statistics.
   - Example Code:
     ```python
     # Grouping by 'City' and calculating mean age
     city_stats = df.groupby('City')['Age'].mean()
     ```
     - Output:
       ```
       City
       Banjarmasin      35.0
       Jakarta         25.0
       Makassar    30.0
       Name: Age, dtype: float64
       ```

8. **Merging and Concatenating:**

   - Explanation: Combining multiple DataFrames based on common columns or indices.
   - Example Code:
     ```python
     # Concatenating vertically
     df_concat = pd.concat([df1, df2])
     ```
     - Output: (Combined DataFrame)
     ```
        Name  Age           City
     0  Eric   25       Jakarta
     1    Robby   30  Makassar
     2 Tina   35    Banjarmasin
     ```
     ```python
     # Merging based on a common column
     merged_df = pd.merge(df1, df2, on='common_column')
     ```
     - Output: (Merged DataFrame)

9. **Applying Functions:**

   - Explanation: Applying custom functions to elements, rows, or columns.
   - Example Code:

     ```python
     # Applying a function to a column
     df['Doubled_Age'] = df['Age'].apply(lambda x: x * 2)
     ```

     - Output:
       ```
        Name  Age           City  Doubled_Age
        0 Eric 25 Jakarta 50
        1 Robby 30 Makassar 60
        2 Tina 35 Banjarmasin 70
       ```

     ```python
      # Applying a function to each element
        df.applymap(lambda x: x.upper() if isinstance(x, str) else x)
     ```

   ```
    - Output: (Applied uppercase to string values)
   ```

```
Name Age City
0 Eric 25 Jakarta
1 Robby 30 Makassar
2 Tina 35 Banjarmasin
```

10. **Exporting Data:**

    - Explanation: Saving DataFrames to different file formats.
    - Example Code:

      ```python
      # Exporting to CSV
      df.to_csv('output.csv', index=False)

      # Exporting to Excel
      df.to_excel('output.xlsx', index=False)
      ```

Pandas is a powerful library that streamlines data manipulation tasks, making it an essential tool in the toolkit of every data scientist and analyst. This cheat sheet serves as a quick reference for common Pandas operations, helping you navigate and manipulate your data efficiently. Whether you're cleaning, exploring, or analyzing data, Pandas has you covered. Happy coding!
