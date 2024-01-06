---
title: "Window Functions"
date: 2023-04-15
tags: ["SQL", "RDBMS"]
draft: false
categories:
  - SQL
thumbnail: ./images/window_functions.png
toc: true
---

# Window Functions Documentation

Window functions are a powerful feature in databases that allow you to perform calculations across a set of rows related to the current row. They are commonly used in SQL queries to analyze and aggregate data within a specific window or range.

## Syntax

The basic syntax for using window functions is as follows:

```sql
SELECT
  column1,
  column2,
  window_function(column3) OVER (PARTITION BY partition_column ORDER BY order_column ROWS BETWEEN N PRECEDING AND M FOLLOWING) AS result_column
FROM
  your_table;
```

- **window_function**: The specific window function to be applied (e.g., SUM, AVG, ROW_NUMBER, etc.).
- **column1, column2, ...**: The columns you want to include in the result set.
- **PARTITION BY partition_column**: Optional clause that divides the result set into partitions to which the window function is applied independently. It is used to group rows based on one or more columns, and the window function is then applied separately to each partition.
- **ORDER BY order_column**: Defines the order in which the rows are processed by the window function. It specifies how the data is sorted within each partition.
- **ROWS BETWEEN N PRECEDING AND M FOLLOWING**: Specifies the window frame, indicating the range of rows used by the window function. It defines the relative position of the current row within the partition.

## Sample Data

Consider the following sample data for better illustration:

**orders** table:

| customer_id | order_date  | order_amount |
|-------------|-------------|--------------|
| 1           | 2023-01-01  | 100          |
| 1           | 2023-02-01  | 150          |
| 2           | 2023-01-15  | 200          |
| 2           | 2023-02-10  | 120          |

## Common Window Functions

### 1. ROW_NUMBER()

Assigns a unique integer to each row within a partition, based on the specified ordering.

```sql
SELECT
  customer_id,
  order_date,
  order_amount,
  ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date) AS row_num
FROM
  orders;
```

**Result:**

| customer_id | order_date  | order_amount | row_num |
|-------------|-------------|--------------|---------|
| 1           | 2023-01-01  | 100          | 1       |
| 1           | 2023-02-01  | 150          | 2       |
| 2           | 2023-01-15  | 200          | 1       |
| 2           | 2023-02-10  | 120          | 2       |

### 2. SUM()

Calculates the sum of a column within a specified window.

```sql
SELECT
  customer_id,
  order_date,
  order_amount,
  SUM(order_amount) OVER (PARTITION BY customer_id ORDER BY order_date) AS running_total
FROM
  orders;
```

**Result:**

| customer_id | order_date  | order_amount | running_total |
|-------------|-------------|--------------|---------------|
| 1           | 2023-01-01  | 100          | 100           |
| 1           | 2023-02-01  | 150          | 250           |
| 2           | 2023-01-15  | 200          | 200           |
| 2           | 2023-02-10  | 120          | 320           |

### 3. AVG()

Computes the average of a column over a specified window.

```sql
SELECT
  customer_id,
  order_date,
  order_amount,
  AVG(order_amount) OVER (PARTITION BY customer_id ORDER BY order_date) AS avg_amount
FROM
  orders;
```

**Result:**

| customer_id | order_date  | order_amount | avg_amount |
|-------------|-------------|--------------|------------|
| 1           | 2023-01-01  | 100          | 100        |
| 1           | 2023-02-01  | 150          | 125        |
| 2           | 2023-01-15  | 200          | 200        |
| 2           | 2023-02-10  | 120          | 160        |

### 4. LEAD() and LAG()

Accesses data from subsequent or preceding rows within the partition.

```sql
SELECT
  customer_id,
  order_date,
  order_amount,
  LEAD(order_amount) OVER (PARTITION BY customer_id ORDER BY order_date) AS next_order_amount,
  LAG(order_amount) OVER (PARTITION BY customer_id ORDER BY order_date) AS prev_order_amount
FROM
  orders;
```

**Result:**

| customer_id | order_date  | order_amount | next_order_amount | prev_order_amount |
|-------------|-------------|--------------|-------------------|-------------------|
| 1           | 2023-01-01  | 100          | 150               | NULL              |
| 1           | 2023-02-01  | 150          | NULL              | 100               |
| 2           | 2023-01-15  | 200          | 120               | NULL              |
| 2           | 2023-02-10  | 120          | NULL              | 200               |

## Conclusion

Window functions are a versatile tool in SQL, providing a way to perform complex analyses over defined windows or partitions. They enhance the capabilities of queries, allowing for efficient and concise data manipulation and analysis. The `OVER` clause, along with `PARTITION BY` and `ORDER BY`, enables fine-grained control over the window within which the window function operates, making it a powerful tool for data analysis.