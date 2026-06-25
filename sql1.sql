CREATE TABLE products (
    product_id VARCHAR(20) PRIMARY KEY,
    product_name TEXT,
    category VARCHAR(100),
    actual_price FLOAT,
    discounted_price FLOAT,
    rating FLOAT
);
CREATE TABLE reviews (
    review_id VARCHAR(50) PRIMARY KEY,
    product_id VARCHAR(20),
    user_id VARCHAR(20),
    rating_count INT,
    review_title TEXT,
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);
INSERT INTO products VALUES
('P1', 'USB Cable', 'Electronics', 500, 199, 4.2),
('P2', 'Mobile Charger', 'Electronics', 800, 349, 4.4),
('P3', 'Earphones', 'Audio', 1500, 799, 4.1);

INSERT INTO reviews VALUES
('R1', 'P1', 'U1', 1200, 'Good product'),
('R2', 'P2', 'U2', 3400, 'Fast charging'),
('R3', 'P3', 'U3', 2100, 'Value for money');

-- select query
select * from products;
select * from reviews;

-- where conditions
SELECT product_name, rating
FROM products
WHERE rating > 4.2;


SELECT category, COUNT(*) AS total_products
FROM products
GROUP BY category;

-- group by
SELECT category, COUNT(*) AS total_products
FROM products
GROUP BY category;

SELECT category, AVG(rating) AS avg_rating
FROM products
GROUP BY category;

-- join 
SELECT p.product_name, r.rating_count
FROM products p
JOIN reviews r
ON p.product_id = r.product_id;

-- join+where
SELECT p.product_name, p.rating, r.rating_count
FROM products p
JOIN reviews r
ON p.product_id = r.product_id
WHERE p.rating > 4.2;

-- groupby + join
SELECT p.category,SUM(r.rating_count) AS total_reviews
FROM products p
JOIN reviews r
ON p.product_id = r.product_id
GROUP BY p.category;
