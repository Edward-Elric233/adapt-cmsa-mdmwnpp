# README

## 概述

该程序根据命令行参数执行不同的评估任务，包括运行实例、检查结果、存储结果以及运行单个实例。

## 使用方法

### 常规选项

1. **`--help` / `-h`**  
   显示帮助信息。  
   **示例**：
   ```bash
   ./program --help
   ```

2. **`--all` / `-a`**  
   等同于同时执行 `--run`、`--check` 和 `--store`。  
   **示例**：
   ```bash
   ./program --all
   ```

---

### 功能选项

1. **`--run` / `-r`**  
   运行所有实例。  
   **示例**：
   ```bash
   ./program --run
   ```

2. **`--check` / `-c`**  
   检查结果是否可用。  
   **示例**：
   ```bash
   ./program --check
   ```

3. **`--store` / `-s`**  
   存储最佳结果到 `results-best`，可选最多 **3 个参数**：
    - `src_results`（默认：`cur`）
    - `dst_results`（默认：`best`）
    - `dst_dir`（自定义目录，默认：空）  
      **示例**：
    - 默认存储：
      ```bash
      ./program --store
      ```
    - 自定义参数：
      ```bash
      ./program --store cur_results best_results custom_dir
      ```

4. **`--single` / `-rs`**  
   运行单个实例，需提供 **5 个参数**：
    - `set_name`、`n`、`m`、`k`、`t`  
      **示例**：
   ```bash
   ./program --single dataset1 10 20 5 100
   ```

---

## 注意事项

1. **参数错误**：
    - `--store` 最多支持 3 个参数。
    - `--single` 必须提供 5 个参数，否则报错。

2. **组合选项**：  
   选项支持组合使用，例如：
   ```bash
   ./program --run --check
   ```