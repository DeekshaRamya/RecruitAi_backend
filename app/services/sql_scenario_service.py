import logging
import random
import time
import httpx
from typing import Dict, Any, List
from fastapi import HTTPException, status
from app.core.config import settings
from app.schemas.assessment import DifficultyDistribution

logger = logging.getLogger("recruitai-backend.sql_scenario_service")

class SqlScenarioService:
    def __init__(self):
        self.api_url = getattr(settings, "SQL_EXECUTION_API_URL", None) or "http://172.176.122.4:5001/execute"
        self.credentials = {
            "host": "172.176.122.4",
            "port": 1433,
            "database": "AdventureWorks",
            "username": "readonly_user",
            "password": "Readonly@123"
        }

    async def execute_sql_via_api(self, query: str, exam_id: str = "sql_scenario_gen") -> Dict[str, Any]:
        """
        Executes a SQL query against the AdventureWorks database via external API at http://172.176.122.4:5001/execute.
        Includes logging of request/response details, proper timeout handling, and retry logic.
        """
        payload = {
            "query": query.strip(),
            "serverType": "sqlserver",
            "credentials": self.credentials,
            "examId": exam_id,
            "userEmail": "system@recruitai.com"
        }

        max_retries = 3
        timeout = httpx.Timeout(15.0)

        logger.info(f"[SqlScenarioService] Calling External SQL Execution API: {self.api_url}")
        logger.info(f"[SqlScenarioService] Request Payload Query: {query.strip()}")

        for attempt in range(1, max_retries + 1):
            try:
                start_time = time.time()
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(self.api_url, json=payload)
                    duration = time.time() - start_time

                logger.info(f"[SqlScenarioService] Response Status: {response.status_code} | Duration: {duration:.2f}s")

                if response.status_code == 200:
                    data = response.json()
                    logger.info(f"[SqlScenarioService] Response Data: rowCount={data.get('rowCount')}, columns={data.get('columns')}, success={data.get('success')}")
                    return data
                else:
                    logger.warning(f"[SqlScenarioService] Attempt {attempt} returned status {response.status_code}: {response.text}")
            except (httpx.TimeoutException, httpx.RequestError) as exc:
                logger.warning(f"[SqlScenarioService] Attempt {attempt} failed with error: {exc}")
                if attempt == max_retries:
                    raise exc

        raise RuntimeError(f"Failed to execute query via AdventureWorks SQL API ({self.api_url}) after {max_retries} attempts.")

    async def generate_sql_scenarios(
        self,
        count: int,
        difficulty_distribution: DifficultyDistribution
    ) -> List[Dict[str, Any]]:
        """
        Generates realistic AdventureWorks Scenario-Based SQL questions exclusively via the external API
        http://172.176.122.4:5001/execute. Enforces strict deduplication across tables, topics, and queries.
        """
        if count <= 0:
            return []

        logger.info(f"[SqlScenarioService] Starting generation of {count} AdventureWorks SQL Scenario questions via API {self.api_url}...")

        # 25+ Curated AdventureWorks scenario templates covering all 5 core schemas & diverse SQL concepts
        scenario_templates = [
            # --- HumanResources Schema ---
            {
                "topic": "Human Resources Employee Demographics",
                "difficulty": "Easy",
                "table": "HumanResources.Employee",
                "scenario": "You are a Junior Data Analyst in the AdventureWorks Human Resources department. HR management requires a quick list of all active salaried employees to analyze organizational demographics and vacation allowances.",
                "question": "Write an SQL query to retrieve the BusinessEntityID, NationalIDNumber, JobTitle, and VacationHours for all salaried employees whose record is currently active.",
                "problemStatement": "Using the 'HumanResources.Employee' table from the AdventureWorks schema (columns: BusinessEntityID, NationalIDNumber, JobTitle, SalariedFlag, CurrentFlag, VacationHours), retrieve all employees where SalariedFlag = 1 and CurrentFlag = 1, ordered descending by VacationHours.",
                "candidateTask": "Write a clean T-SQL SELECT query against HumanResources.Employee filtering by SalariedFlag = 1 and CurrentFlag = 1, sorting by VacationHours DESC.",
                "query": "SELECT TOP 5 BusinessEntityID, NationalIDNumber, JobTitle, VacationHours FROM HumanResources.Employee WHERE SalariedFlag = 1 AND CurrentFlag = 1 ORDER BY VacationHours DESC;",
                "evaluationCriteria": "Correct schema name HumanResources.Employee, accurate WHERE clause on SalariedFlag and CurrentFlag, proper ORDER BY clause.",
                "explanation": "Filtering by SalariedFlag = 1 and CurrentFlag = 1 isolates active salaried staff, while ORDER BY VacationHours DESC lists employees with the highest leave accumulation first."
            },
            {
                "topic": "Department Staffing Aggregations",
                "difficulty": "Medium",
                "table": "HumanResources.Department",
                "scenario": "The HR Director is conducting an organizational workload audit to identify departments with high headcount density.",
                "question": "Write an SQL query to list the DepartmentName and current EmployeeCount for all departments that currently have more than 5 active assignment records.",
                "problemStatement": "Join 'HumanResources.Department' (d) with 'HumanResources.EmployeeDepartmentHistory' (edh) on DepartmentID where edh.EndDate IS NULL. Group by d.Name and filter using HAVING COUNT(edh.BusinessEntityID) > 5, ordered descending by EmployeeCount.",
                "candidateTask": "Write an INNER JOIN query grouping by department name and applying a HAVING clause for count greater than 5.",
                "query": "SELECT TOP 5 d.Name AS DepartmentName, COUNT(edh.BusinessEntityID) AS EmployeeCount FROM HumanResources.Department d INNER JOIN HumanResources.EmployeeDepartmentHistory edh ON d.DepartmentID = edh.DepartmentID WHERE edh.EndDate IS NULL GROUP BY d.Name HAVING COUNT(edh.BusinessEntityID) > 5 ORDER BY EmployeeCount DESC;",
                "evaluationCriteria": "Valid INNER JOIN syntax, EndDate IS NULL filter for current assignments, correct GROUP BY and HAVING clauses.",
                "explanation": "Joining Department and EmployeeDepartmentHistory isolates active department headcounts, while HAVING filters for major teams."
            },
            {
                "topic": "Company Work Shift Schedules",
                "difficulty": "Easy",
                "table": "HumanResources.Shift",
                "scenario": "Logistics managers need a reference list of operating work shifts to schedule plant maintenance and security rotas.",
                "question": "Write an SQL query to display ShiftID, Name, StartTime, and EndTime for all shift schedules ordered by StartTime.",
                "problemStatement": "Select ShiftID, Name, StartTime, EndTime from 'HumanResources.Shift', ordering the results chronologically by StartTime.",
                "candidateTask": "Write a T-SQL SELECT query against HumanResources.Shift ordered by StartTime ASC.",
                "query": "SELECT ShiftID, Name, StartTime, EndTime FROM HumanResources.Shift ORDER BY StartTime ASC;",
                "evaluationCriteria": "Correct table reference HumanResources.Shift and accurate ORDER BY StartTime.",
                "explanation": "Sorting shifts by StartTime provides an ordered view of daily operational rosters."
            },
            {
                "topic": "Employee Department History Tracking",
                "difficulty": "Hard",
                "table": "HumanResources.EmployeeDepartmentHistory",
                "scenario": "HR Compliance needs to track recent departmental assignment starts for auditing internal job movements.",
                "question": "Write an SQL query to retrieve BusinessEntityID, JobTitle, DepartmentName, and StartDate for active department assignments ordered by StartDate descending.",
                "problemStatement": "Join 'HumanResources.EmployeeDepartmentHistory' (edh) with 'HumanResources.Employee' (e) and 'HumanResources.Department' (d) where edh.EndDate IS NULL, ordered by edh.StartDate DESC.",
                "candidateTask": "Write a 3-table INNER JOIN query retrieving active employee department assignments.",
                "query": "SELECT TOP 5 edh.BusinessEntityID, e.JobTitle, d.Name AS DepartmentName, edh.StartDate FROM HumanResources.EmployeeDepartmentHistory edh INNER JOIN HumanResources.Employee e ON edh.BusinessEntityID = e.BusinessEntityID INNER JOIN HumanResources.Department d ON edh.DepartmentID = d.DepartmentID WHERE edh.EndDate IS NULL ORDER BY edh.StartDate DESC;",
                "evaluationCriteria": "Correct multi-table INNER JOIN syntax across edh, e, and d, with EndDate IS NULL filter.",
                "explanation": "Multi-table join presents comprehensive assignment details including job titles and department names."
            },

            # --- Sales Schema ---
            {
                "topic": "Sales Order Financial Overview",
                "difficulty": "Easy",
                "table": "Sales.SalesOrderHeader",
                "scenario": "The Sales Operations Lead at AdventureWorks requires an overview of recent high-value sales transactions to evaluate revenue trends for key accounts.",
                "question": "Write an SQL query to retrieve the SalesOrderID, OrderDate, CustomerID, and TotalDue for all sales orders where TotalDue exceeds $10,000.",
                "problemStatement": "Using the 'Sales.SalesOrderHeader' table (columns: SalesOrderID, OrderDate, CustomerID, TotalDue), filter orders where TotalDue > 10000 and order the output by OrderDate in descending order.",
                "candidateTask": "Write a T-SQL SELECT query against Sales.SalesOrderHeader filtering TotalDue > 10000, ordered by OrderDate DESC.",
                "query": "SELECT TOP 5 SalesOrderID, OrderDate, CustomerID, TotalDue FROM Sales.SalesOrderHeader WHERE TotalDue > 10000 ORDER BY OrderDate DESC;",
                "evaluationCriteria": "Valid table reference Sales.SalesOrderHeader, WHERE filter on TotalDue > 10000, and ORDER BY OrderDate DESC.",
                "explanation": "Selecting top transactions over $10,000 isolates high-value deals for executive financial review."
            },
            {
                "topic": "Customer Account Audit & Outer Join",
                "difficulty": "Medium",
                "table": "Sales.Customer",
                "scenario": "Sales Account Managers want to identify inactive customer accounts that have never placed an order in the database.",
                "question": "Write an SQL query using a LEFT JOIN to find CustomerID, AccountNumber, and StoreID for customers who have no recorded sales orders in Sales.SalesOrderHeader.",
                "problemStatement": "Perform a LEFT JOIN between 'Sales.Customer' (c) and 'Sales.SalesOrderHeader' (soh) on CustomerID. Filter for records where soh.SalesOrderID IS NULL.",
                "candidateTask": "Write a LEFT JOIN query checking for NULL values in the joined table key.",
                "query": "SELECT TOP 5 c.CustomerID, c.AccountNumber, c.StoreID FROM Sales.Customer c LEFT JOIN Sales.SalesOrderHeader soh ON c.CustomerID = soh.CustomerID WHERE soh.SalesOrderID IS NULL ORDER BY c.CustomerID ASC;",
                "evaluationCriteria": "Correct LEFT JOIN on CustomerID and accurate WHERE soh.SalesOrderID IS NULL filter.",
                "explanation": "LEFT JOIN with IS NULL filter isolates unengaged customer accounts for targeted retention campaigns."
            },
            {
                "topic": "Sales Performance Leaderboard CTE",
                "difficulty": "Hard",
                "table": "Sales.SalesPerson",
                "scenario": "Executive Leadership wants a leaderboard ranking sales representatives based on year-to-date sales figures.",
                "question": "Write an SQL query using a CTE and ROW_NUMBER() window function to rank SalesPerson records by SalesYTD descending.",
                "problemStatement": "Construct a Common Table Expression (CTE) named 'RankedSales' using ROW_NUMBER() OVER (ORDER BY SalesYTD DESC) AS RankNum from 'Sales.SalesPerson' where SalesYTD IS NOT NULL. Select BusinessEntityID, SalesQuota, SalesYTD, and RankNum where RankNum <= 5.",
                "candidateTask": "Write a CTE with ROW_NUMBER() window function ranking sales reps by SalesYTD.",
                "query": "WITH RankedSales AS (SELECT BusinessEntityID, SalesQuota, SalesYTD, ROW_NUMBER() OVER (ORDER BY SalesYTD DESC) AS RankNum FROM Sales.SalesPerson WHERE SalesYTD IS NOT NULL) SELECT TOP 5 BusinessEntityID, SalesQuota, SalesYTD, RankNum FROM RankedSales WHERE RankNum <= 5;",
                "evaluationCriteria": "Valid WITH CTE clause, proper ROW_NUMBER() OVER (ORDER BY ...) window function syntax.",
                "explanation": "CTE and window functions cleanly rank top sales performers without requiring subqueries."
            },
            {
                "topic": "Product Revenue Summarization",
                "difficulty": "Hard",
                "table": "Sales.SalesOrderDetail",
                "scenario": "Product Managers need to know which products have generated total sales revenue exceeding $50,000 across all completed order items.",
                "question": "Write an SQL query to group 'Sales.SalesOrderDetail' by ProductID, calculating TotalQuantitySold (SUM of OrderQty) and RevenueGenerated (SUM of LineTotal), filtering for RevenueGenerated > 50000.",
                "problemStatement": "Group 'Sales.SalesOrderDetail' by ProductID. Use SUM(OrderQty) and SUM(LineTotal). Apply HAVING SUM(LineTotal) > 50000, sorted descending by RevenueGenerated.",
                "candidateTask": "Write a GROUP BY query with aggregate SUM functions and HAVING clause filtering revenue over 50,000.",
                "query": "SELECT TOP 5 ProductID, SUM(OrderQty) AS TotalQuantitySold, SUM(LineTotal) AS RevenueGenerated FROM Sales.SalesOrderDetail GROUP BY ProductID HAVING SUM(LineTotal) > 50000 ORDER BY RevenueGenerated DESC;",
                "evaluationCriteria": "Proper GROUP BY ProductID, SUM aggregates, and HAVING clause filtering LineTotal > 50000.",
                "explanation": "GROUP BY with HAVING aggregates itemized sales line totals to identify top grossing product lines."
            },
            {
                "topic": "Sales Territory Revenue Comparison",
                "difficulty": "Medium",
                "table": "Sales.SalesTerritory",
                "scenario": "Regional Vice Presidents are reviewing territory sales growth to compare current year performance against last year.",
                "question": "Write an SQL query to retrieve TerritoryID, Name, [Group], SalesYTD, and SalesLastYear for territories where SalesYTD is greater than SalesLastYear.",
                "problemStatement": "Select territory details from 'Sales.SalesTerritory' where SalesYTD > SalesLastYear, ordered by SalesYTD DESC.",
                "candidateTask": "Write a T-SQL query comparing two numerical columns in Sales.SalesTerritory.",
                "query": "SELECT TerritoryID, Name, [Group], SalesYTD, SalesLastYear FROM Sales.SalesTerritory WHERE SalesYTD > SalesLastYear ORDER BY SalesYTD DESC;",
                "evaluationCriteria": "Correct table reference Sales.SalesTerritory and valid comparison filter SalesYTD > SalesLastYear.",
                "explanation": "Comparing YTD against historical sales highlights growing sales territories."
            },
            {
                "topic": "Promotional Discount Special Offers",
                "difficulty": "Medium",
                "table": "Sales.SpecialOffer",
                "scenario": "Marketing Analysts are reviewing historical promotional campaigns to assess high-discount special offers.",
                "question": "Write an SQL query to select SpecialOfferID, Description, DiscountPct, Type, and Category for special offers with a DiscountPct greater than 0.10.",
                "problemStatement": "Select from 'Sales.SpecialOffer' where DiscountPct > 0.10, ordered by DiscountPct DESC.",
                "candidateTask": "Write a SELECT query filtering decimal percentage values in Sales.SpecialOffer.",
                "query": "SELECT TOP 5 SpecialOfferID, Description, DiscountPct, Type, Category FROM Sales.SpecialOffer WHERE DiscountPct > 0.10 ORDER BY DiscountPct DESC;",
                "evaluationCriteria": "Correct table reference Sales.SpecialOffer and valid filter DiscountPct > 0.10.",
                "explanation": "Filtering discounts over 10% isolates significant promotional price markdowns."
            },
            {
                "topic": "Store Sales Performance Join",
                "difficulty": "Medium",
                "table": "Sales.Store",
                "scenario": "Retail Account Management wants to inspect store client accounts linked with top-performing sales representatives.",
                "question": "Write an SQL query joining 'Sales.Store' (s) and 'Sales.SalesPerson' (sp) on SalesPersonID to display StoreName and SalesYTD.",
                "problemStatement": "Join Sales.Store with Sales.SalesPerson on SalesPersonID = BusinessEntityID, selecting Store BusinessEntityID, Store Name, and SalesYTD ordered by SalesYTD DESC.",
                "candidateTask": "Write an INNER JOIN query connecting retail stores to assigned sales representatives.",
                "query": "SELECT TOP 5 s.BusinessEntityID, s.Name AS StoreName, sp.SalesYTD FROM Sales.Store s INNER JOIN Sales.SalesPerson sp ON s.SalesPersonID = sp.BusinessEntityID ORDER BY sp.SalesYTD DESC;",
                "evaluationCriteria": "Valid INNER JOIN between Store and SalesPerson on key SalesPersonID = BusinessEntityID.",
                "explanation": "Joining retail store accounts with sales reps evaluates channel distribution performance."
            },

            # --- Production Schema ---
            {
                "topic": "Product Catalog & Pricing Analysis",
                "difficulty": "Medium",
                "table": "Production.Product",
                "scenario": "AdventureWorks Product Management is auditing the commercial catalog to identify premium manufactured items and ensure accurate price positioning.",
                "question": "Write an SQL query to select ProductID, Name, ProductNumber, and ListPrice for all products with a ListPrice between $500 and $2,000.",
                "problemStatement": "Using the 'Production.Product' table (columns: ProductID, Name, ProductNumber, ListPrice, FinishedGoodsFlag), select active products where ListPrice BETWEEN 500 AND 2000 and FinishedGoodsFlag = 1, ordered by ListPrice DESC.",
                "candidateTask": "Write a T-SQL SELECT query against Production.Product filtering by ListPrice BETWEEN 500 AND 2000 and FinishedGoodsFlag = 1.",
                "query": "SELECT TOP 5 ProductID, Name, ProductNumber, ListPrice FROM Production.Product WHERE ListPrice BETWEEN 500 AND 2000 AND FinishedGoodsFlag = 1 ORDER BY ListPrice DESC;",
                "evaluationCriteria": "Correct usage of Production.Product table, valid BETWEEN clause or range comparison on ListPrice, and proper ordering.",
                "explanation": "Filtering between $500 and $2,000 isolates mid-tier to premium finished goods for inventory strategy."
            },
            {
                "topic": "Product Category Pricing Multi-Join",
                "difficulty": "Hard",
                "table": "Production.ProductCategory",
                "scenario": "Merchandising Strategy needs a category-level price audit summarizing product count and average list price across top product categories.",
                "question": "Write an SQL query joining Production.Product (p), Production.ProductSubcategory (ps), and Production.ProductCategory (pc) to calculate TotalProducts and AvgPrice per category name.",
                "problemStatement": "Join Product, ProductSubcategory, and ProductCategory. Group by pc.Name, selecting pc.Name AS CategoryName, COUNT(p.ProductID) AS TotalProducts, and AVG(p.ListPrice) AS AvgPrice, ordered by AvgPrice DESC.",
                "candidateTask": "Write a 3-table INNER JOIN with GROUP BY and aggregate functions COUNT and AVG.",
                "query": "SELECT TOP 5 pc.Name AS CategoryName, COUNT(p.ProductID) AS TotalProducts, AVG(p.ListPrice) AS AvgPrice FROM Production.Product p INNER JOIN Production.ProductSubcategory ps ON p.ProductSubcategoryID = ps.ProductSubcategoryID INNER JOIN Production.ProductCategory pc ON ps.ProductCategoryID = pc.ProductCategoryID GROUP BY pc.Name ORDER BY AvgPrice DESC;",
                "evaluationCriteria": "Correct 3-table JOIN hierarchy (Product -> Subcategory -> Category), proper GROUP BY pc.Name, and valid COUNT / AVG functions.",
                "explanation": "Multi-table join summarizes item prices at the top category level for executive pricing strategy."
            },
            {
                "topic": "Work Order Processing Lead Time Date Math",
                "difficulty": "Medium",
                "table": "Production.WorkOrder",
                "scenario": "Manufacturing Engineers are analyzing production line bottlenecks by calculating total processing days for work orders.",
                "question": "Write an SQL query using DATEDIFF() to calculate ProcessingDays between StartDate and EndDate for completed work orders taking longer than 5 days.",
                "problemStatement": "From 'Production.WorkOrder', select WorkOrderID, ProductID, OrderQty, and DATEDIFF(day, StartDate, EndDate) AS ProcessingDays where EndDate IS NOT NULL and DATEDIFF(day, StartDate, EndDate) > 5, ordered by ProcessingDays DESC.",
                "candidateTask": "Write a T-SQL query using the DATEDIFF function to calculate date differences.",
                "query": "SELECT TOP 5 WorkOrderID, ProductID, OrderQty, DATEDIFF(day, StartDate, EndDate) AS ProcessingDays FROM Production.WorkOrder WHERE EndDate IS NOT NULL AND DATEDIFF(day, StartDate, EndDate) > 5 ORDER BY ProcessingDays DESC;",
                "evaluationCriteria": "Correct usage of DATEDIFF(day, StartDate, EndDate) function and proper filter logic.",
                "explanation": "Using DATEDIFF calculates operational manufacturing turnaround duration."
            },
            {
                "topic": "Manufacturing Plant Location Costs",
                "difficulty": "Easy",
                "table": "Production.Location",
                "scenario": "Cost Accounting wants to review manufacturing facility locations with high hourly labor cost rates.",
                "question": "Write an SQL query to retrieve LocationID, Name, CostRate, and Availability for locations with CostRate exceeding $10.00.",
                "problemStatement": "Select from 'Production.Location' where CostRate > 10, ordered by CostRate DESC.",
                "candidateTask": "Write a SELECT query filtering numerical cost rate values in Production.Location.",
                "query": "SELECT TOP 5 LocationID, Name, CostRate, Availability FROM Production.Location WHERE CostRate > 10 ORDER BY CostRate DESC;",
                "evaluationCriteria": "Correct table reference Production.Location and accurate WHERE CostRate > 10 filter.",
                "explanation": "Filtering locations by CostRate identifies high-overhead manufacturing centers."
            },
            {
                "topic": "Bill of Materials Component Assembly",
                "difficulty": "Medium",
                "table": "Production.BillOfMaterials",
                "scenario": "Assembly Plant Supervisors are reviewing active product bills of materials to ensure accurate sub-assembly component quantities.",
                "question": "Write an SQL query joining Production.BillOfMaterials (bom) with Production.Product (p) on ComponentID = ProductID to display active assembly components.",
                "problemStatement": "Join BillOfMaterials with Product on ComponentID = ProductID. Select BillOfMaterialsID, ComponentName (p.Name), and PerAssemblyQty where bom.EndDate IS NULL, ordered by PerAssemblyQty DESC.",
                "candidateTask": "Write an INNER JOIN query between BOM and Product for active components.",
                "query": "SELECT TOP 5 bom.BillOfMaterialsID, p.Name AS ComponentName, bom.PerAssemblyQty FROM Production.BillOfMaterials bom INNER JOIN Production.Product p ON bom.ComponentID = p.ProductID WHERE bom.EndDate IS NULL ORDER BY bom.PerAssemblyQty DESC;",
                "evaluationCriteria": "Correct INNER JOIN on ComponentID = ProductID and EndDate IS NULL filter.",
                "explanation": "Joining BOM with Product retrieves human-readable names for sub-assembly components."
            },
            {
                "topic": "Inventory Transaction History Volume",
                "difficulty": "Hard",
                "table": "Production.TransactionHistory",
                "scenario": "Warehouse Operations needs a summary of high-volume material movements grouped by ProductID and TransactionType.",
                "question": "Write an SQL query to group 'Production.TransactionHistory' by ProductID and TransactionType, calculating TotalQty (SUM of Quantity) for product transactions over 1,000 units.",
                "problemStatement": "Group Production.TransactionHistory by ProductID, TransactionType. Calculate SUM(Quantity) AS TotalQty with HAVING SUM(Quantity) > 1000, ordered by TotalQty DESC.",
                "candidateTask": "Write a GROUP BY query with multi-column grouping and HAVING filter on aggregate sum.",
                "query": "SELECT TOP 5 ProductID, TransactionType, SUM(Quantity) AS TotalQty FROM Production.TransactionHistory GROUP BY ProductID, TransactionType HAVING SUM(Quantity) > 1000 ORDER BY TotalQty DESC;",
                "evaluationCriteria": "Correct GROUP BY ProductID, TransactionType and valid HAVING SUM(Quantity) > 1000 clause.",
                "explanation": "Multi-column grouping isolates large scale material movements across inventory transaction types."
            },

            # --- Purchasing Schema ---
            {
                "topic": "Purchasing & Vendor Management",
                "difficulty": "Hard",
                "table": "Purchasing.PurchaseOrderHeader",
                "scenario": "The Supply Chain Audit Committee requires an evaluation of purchase order commitments to manage supplier cash flow and review pending logistics.",
                "question": "Write an SQL query to retrieve PurchaseOrderID, VendorID, OrderDate, and SubTotal for purchase orders with Status equal to 4 (Complete) and SubTotal greater than $50,000.",
                "problemStatement": "Using the 'Purchasing.PurchaseOrderHeader' table (columns: PurchaseOrderID, VendorID, OrderDate, Status, SubTotal), select completed purchase orders (Status = 4) where SubTotal > 50000, sorted descending by SubTotal.",
                "candidateTask": "Write a T-SQL SELECT query against Purchasing.PurchaseOrderHeader filtering Status = 4 and SubTotal > 50000.",
                "query": "SELECT TOP 5 PurchaseOrderID, VendorID, OrderDate, SubTotal FROM Purchasing.PurchaseOrderHeader WHERE Status = 4 AND SubTotal > 50000 ORDER BY SubTotal DESC;",
                "evaluationCriteria": "Correct table name Purchasing.PurchaseOrderHeader, valid multi-condition WHERE clause, and descending sorting.",
                "explanation": "Filtering for Status = 4 and SubTotal > $50,000 focuses audit attention on high-capital completed supply purchases."
            },
            {
                "topic": "Vendor Spend Analysis & Aggregations",
                "difficulty": "Medium",
                "table": "Purchasing.Vendor",
                "scenario": "Procurement Specialists are identifying top preferred suppliers based on cumulative order expenditure.",
                "question": "Write an SQL query joining Purchasing.Vendor (v) and Purchasing.PurchaseOrderHeader (poh) to calculate TotalOrders and TotalSpent per vendor, filtering for TotalSpent > $100,000.",
                "problemStatement": "Join Vendor with PurchaseOrderHeader on VendorID. Group by v.VendorID, v.Name. Calculate COUNT(poh.PurchaseOrderID) AS TotalOrders and SUM(poh.TotalDue) AS TotalSpent. Filter HAVING SUM(poh.TotalDue) > 100000.",
                "candidateTask": "Write an INNER JOIN query with GROUP BY vendor details and HAVING aggregate sum condition.",
                "query": "SELECT TOP 5 v.VendorID, v.Name AS VendorName, COUNT(poh.PurchaseOrderID) AS TotalOrders, SUM(poh.TotalDue) AS TotalSpent FROM Purchasing.Vendor v INNER JOIN Purchasing.PurchaseOrderHeader poh ON v.VendorID = poh.VendorID GROUP BY v.VendorID, v.Name HAVING SUM(poh.TotalDue) > 100000 ORDER BY TotalSpent DESC;",
                "evaluationCriteria": "Correct INNER JOIN on VendorID, GROUP BY v.VendorID, v.Name, and HAVING clause on SUM(TotalDue).",
                "explanation": "Joining Vendors with order headers aggregates supplier expenditure for strategic sourcing negotiations."
            },
            {
                "topic": "Purchase Order Delivery Status CASE Expression",
                "difficulty": "Hard",
                "table": "Purchasing.PurchaseOrderDetail",
                "scenario": "Logistics Coordinators need a categorized report of order item fulfillment status using conditional logic.",
                "question": "Write an SQL query using a CASE statement on 'Purchasing.PurchaseOrderDetail' to output a DeliveryStatus column ('Fully Received', 'Partial Receive', or 'Over Received').",
                "problemStatement": "Select PurchaseOrderID, ProductID, OrderQty, UnitPrice, and a CASE expression comparing ReceivedQty to OrderQty (WHEN ReceivedQty = OrderQty THEN 'Fully Received' WHEN ReceivedQty < OrderQty THEN 'Partial Receive' ELSE 'Over Received' END) AS DeliveryStatus for OrderQty > 100.",
                "candidateTask": "Write a SELECT query with a conditional CASE WHEN expression evaluating received vs ordered quantities.",
                "query": "SELECT TOP 5 PurchaseOrderID, ProductID, OrderQty, UnitPrice, CASE WHEN ReceivedQty = OrderQty THEN 'Fully Received' WHEN ReceivedQty < OrderQty THEN 'Partial Receive' ELSE 'Over Received' END AS DeliveryStatus FROM Purchasing.PurchaseOrderDetail WHERE OrderQty > 100 ORDER BY PurchaseOrderID DESC;",
                "evaluationCriteria": "Valid CASE WHEN ... THEN ... ELSE ... END syntax and logical quantity comparison.",
                "explanation": "CASE expressions classify complex inventory receiving statuses into actionable category labels."
            },
            {
                "topic": "Shipment Freight Rates Comparison",
                "difficulty": "Easy",
                "table": "Purchasing.ShipMethod",
                "scenario": "Logistics Dispatchers are reviewing freight shipping service rates to minimize transport costs.",
                "question": "Write an SQL query to display ShipMethodID, Name, ShipBase, and ShipRate from 'Purchasing.ShipMethod' ordered by ShipRate ascending.",
                "problemStatement": "Select all shipping methods from 'Purchasing.ShipMethod' ordered by ShipRate ASC.",
                "candidateTask": "Write a SELECT query ordering freight records in Purchasing.ShipMethod.",
                "query": "SELECT ShipMethodID, Name, ShipBase, ShipRate FROM Purchasing.ShipMethod ORDER BY ShipRate ASC;",
                "evaluationCriteria": "Correct table reference Purchasing.ShipMethod and ORDER BY ShipRate ASC.",
                "explanation": "Sorting shipping methods by ShipRate enables quick selection of economical freight providers."
            },

            # --- Person Schema ---
            {
                "topic": "Customer Information Directory",
                "difficulty": "Medium",
                "table": "Person.Person",
                "scenario": "The Customer Relationship Management team wants to isolate specific customer contact details for promotional outreach.",
                "question": "Write an SQL query to retrieve BusinessEntityID, PersonType, FirstName, and LastName for all individuals where PersonType is 'SC' (Store Contact) or 'IN' (Individual Customer).",
                "problemStatement": "Using the 'Person.Person' table (columns: BusinessEntityID, PersonType, FirstName, LastName), retrieve records where PersonType IN ('SC', 'IN'), ordered by LastName and FirstName.",
                "candidateTask": "Write a T-SQL SELECT query against Person.Person filtering PersonType IN ('SC', 'IN') and sorting alphabetically.",
                "query": "SELECT TOP 5 BusinessEntityID, PersonType, FirstName, LastName FROM Person.Person WHERE PersonType IN ('SC', 'IN') ORDER BY LastName, FirstName;",
                "evaluationCriteria": "Correct table reference Person.Person, accurate IN operator or OR condition for PersonType, and proper double sorting.",
                "explanation": "Using IN ('SC', 'IN') targets store representatives and individual retail consumers for direct marketing."
            },
            {
                "topic": "Geographic City Density Analysis",
                "difficulty": "Medium",
                "table": "Person.Address",
                "scenario": "Regional Marketing Planners are evaluating geographic customer distribution to determine target cities for new retail outlets.",
                "question": "Write an SQL query grouping 'Person.Address' by City, counting AddressID as AddressCount, and filtering for cities with at least 10 recorded addresses.",
                "problemStatement": "Group Person.Address by City. Select City, COUNT(AddressID) AS AddressCount with HAVING COUNT(AddressID) >= 10, ordered by AddressCount DESC.",
                "candidateTask": "Write a GROUP BY query on City with HAVING COUNT >= 10.",
                "query": "SELECT TOP 5 City, COUNT(AddressID) AS AddressCount FROM Person.Address GROUP BY City HAVING COUNT(AddressID) >= 10 ORDER BY AddressCount DESC;",
                "evaluationCriteria": "Correct GROUP BY City syntax and HAVING COUNT(AddressID) >= 10 filter.",
                "explanation": "Grouping customer addresses by city highlights key metropolitan customer density zones."
            },
            {
                "topic": "Corporate Email Directory String Matching",
                "difficulty": "Easy",
                "table": "Person.EmailAddress",
                "scenario": "IT Communications is compiling an internal email address directory for corporate personnel.",
                "question": "Write an SQL query joining Person.Person (p) and Person.EmailAddress (e) to retrieve FirstName, LastName, and EmailAddress filtering for domain 'adventure-works.com'.",
                "problemStatement": "Join Person.Person with Person.EmailAddress on BusinessEntityID. Filter where e.EmailAddress LIKE '%adventure-works.com%', ordered by LastName, FirstName.",
                "candidateTask": "Write an INNER JOIN query using LIKE pattern matching on email strings.",
                "query": "SELECT TOP 5 p.FirstName, p.LastName, e.EmailAddress FROM Person.Person p INNER JOIN Person.EmailAddress e ON p.BusinessEntityID = e.BusinessEntityID WHERE e.EmailAddress LIKE '%adventure-works.com%' ORDER BY p.LastName, p.FirstName;",
                "evaluationCriteria": "Valid INNER JOIN on BusinessEntityID and correct LIKE '%adventure-works.com%' wildcards.",
                "explanation": "Joining Person with EmailAddress and filtering with LIKE retrieves official corporate contact details."
            },
            {
                "topic": "State & Country Region Mapping Join",
                "difficulty": "Easy",
                "table": "Person.StateProvince",
                "scenario": "Global Logistics needs a standardized list mapping state/province codes to their respective country region names.",
                "question": "Write an SQL query joining Person.StateProvince (sp) with Person.CountryRegion (cr) on CountryRegionCode to display StateCode, StateName, and CountryName.",
                "problemStatement": "Join Person.StateProvince with Person.CountryRegion on CountryRegionCode. Select sp.StateProvinceCode, sp.Name AS StateName, cr.Name AS CountryName, ordered by sp.Name ASC.",
                "candidateTask": "Write an INNER JOIN query mapping states to countries.",
                "query": "SELECT TOP 5 sp.StateProvinceCode, sp.Name AS StateName, cr.Name AS CountryName FROM Person.StateProvince sp INNER JOIN Person.CountryRegion cr ON sp.CountryRegionCode = cr.CountryRegionCode ORDER BY sp.Name ASC;",
                "evaluationCriteria": "Correct INNER JOIN on CountryRegionCode and accurate column alias definitions.",
                "explanation": "Joining StateProvince with CountryRegion maps sub-national administrative divisions to sovereign countries."
            }
        ]

        # Target difficulty distribution calculations
        easy_needed = round(count * (difficulty_distribution.easy / 100.0))
        medium_needed = round(count * (difficulty_distribution.medium / 100.0))
        hard_needed = count - (easy_needed + medium_needed)

        target_difficulties = (
            ["Easy"] * easy_needed +
            ["Medium"] * medium_needed +
            ["Hard"] * hard_needed
        )
        if len(target_difficulties) < count:
            target_difficulties.extend(["Medium"] * (count - len(target_difficulties)))

        # Tracking sets to ensure 100% uniqueness per assessment generation
        used_tables = set()
        used_topics = set()
        used_queries = set()

        generated_questions = []

        for i in range(count):
            diff = target_difficulties[i] if i < len(target_difficulties) else "Medium"

            # Filter candidates by difficulty
            matching_templates = [t for t in scenario_templates if t["difficulty"] == diff]
            if not matching_templates:
                matching_templates = scenario_templates

            # Find candidates whose table, topic, AND query haven't been used yet
            unused_candidates = [
                t for t in matching_templates
                if t["table"] not in used_tables and t["topic"] not in used_topics and t["query"] not in used_queries
            ]

            if not unused_candidates:
                # Fallback: find unused tables across all difficulty tiers
                unused_candidates = [
                    t for t in scenario_templates
                    if t["table"] not in used_tables and t["query"] not in used_queries
                ]

            if not unused_candidates:
                # Final fallback if pool exhausted
                unused_candidates = matching_templates

            # Shuffle candidates to ensure variety across calls
            random.shuffle(unused_candidates)

            success = False
            for tmpl in unused_candidates:
                target_table = tmpl["table"]
                solution_query = tmpl["query"]

                try:
                    # 1. Execute solution query against external API at http://172.176.122.4:5001/execute
                    logger.info(f"[SqlScenarioService] Executing Solution Query against API {self.api_url}: {solution_query}")
                    api_result = await self.execute_sql_via_api(query=solution_query, exam_id=f"scenario_{i+1}")

                    columns = api_result.get("columns", [])
                    rows = api_result.get("rows", [])

                    # 2. Format tabular expected output from actual API results
                    markdown_table = self._format_rows_to_markdown_table(columns, rows)

                    # 3. Retrieve sample data via external API for table DDL & sample rows
                    sample_query = f"SELECT TOP 5 * FROM {target_table};"
                    logger.info(f"[SqlScenarioService] Fetching Sample Data via API {self.api_url}: {sample_query}")
                    sample_api_result = await self.execute_sql_via_api(query=sample_query, exam_id=f"sample_{i+1}")

                    sample_rows = sample_api_result.get("rows", [])
                    sample_cols = sample_api_result.get("columns", [])

                    sample_data_lines = [f"-- Real data retrieved dynamically from connected SQL Server ({target_table})"]
                    for s_row in sample_rows[:5]:
                        vals = []
                        for c in sample_cols:
                            val = s_row.get(c)
                            if val is None:
                                vals.append("NULL")
                            elif isinstance(val, (int, float)):
                                vals.append(str(val))
                            else:
                                clean_val = str(val).replace("'", "''")
                                vals.append(f"'{clean_val}'")
                        sample_data_lines.append(f"INSERT INTO {target_table} VALUES ({', '.join(vals)});")

                    schema_ddl = [f"-- Live SQL Server Schema\nCREATE TABLE {target_table} ({', '.join(sample_cols)} );"]

                    q_obj = {
                        "id": f"q_sql_scenario_{i+1}_{int(time.time())}",
                        "subject": "SQL",
                        "topic": tmpl["topic"],
                        "type": "SCENARIO",
                        "difficulty": diff,
                        "scenario": tmpl["scenario"],
                        "question": tmpl["question"],
                        "problemStatement": tmpl["problemStatement"],
                        "candidateTask": tmpl["candidateTask"],
                        "expectedAnswer": solution_query,
                        "correctAnswer": solution_query,
                        "starterCode": f"-- Write your T-SQL query here against the live AdventureWorks database\nSELECT * FROM {target_table};",
                        "starter_code": f"-- Write your T-SQL query here against the live AdventureWorks database\nSELECT * FROM {target_table};",
                        "evaluationCriteria": tmpl["evaluationCriteria"],
                        "explanation": tmpl["explanation"],
                        "databaseSchema": schema_ddl,
                        "sampleData": sample_data_lines,
                        "exampleOutput": markdown_table,
                        "expectedOutput": markdown_table
                    }

                    used_tables.add(tmpl["table"])
                    used_topics.add(tmpl["topic"])
                    used_queries.add(tmpl["query"])

                    generated_questions.append(q_obj)
                    logger.info(f"[SqlScenarioService] Successfully generated SQL Scenario question {i+1} using {target_table}.")
                    success = True
                    break

                except Exception as ex:
                    logger.warning(f"[SqlScenarioService] Attempt for template {tmpl['table']} failed: {ex}. Trying next candidate template...")
                    continue

            if not success:
                logger.error(f"[SqlScenarioService] Failed to generate a valid SQL scenario question for slot {i+1}.")
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Failed to connect to AdventureWorks SQL Execution API (http://172.176.122.4:5001/execute)."
                )

        return generated_questions

    @staticmethod
    def _format_rows_to_markdown_table(columns: list, rows: list, max_rows: int = 5) -> str:
        if not columns:
            return "No records found."
        lines = []
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
        display_rows = rows[:max_rows]
        for row in display_rows:
            row_vals = [str(row.get(col, "")) if row.get(col) is not None else "NULL" for col in columns]
            lines.append("| " + " | ".join(row_vals) + " |")
        if len(rows) > max_rows:
            lines.append(f"*(showing top {max_rows} of {len(rows)} returned records)*")
        return "\n".join(lines)
