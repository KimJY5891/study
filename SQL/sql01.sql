-- #1
-- : 주석 내용을 작성하세요

-- #2
SELECT * FROM Customers;
SELECT * -- 가져오겠다.(SELECT) 모든 컬럼을(*)

-- #3
SELECT * FROM Customers -- FROM (테이블 네임)
-- 즉, Customers 라는 테이블에서(FROM), 모든 컬럼(*)을 선택(SELECT)해서 가져와라!

-- #4
SELECT CustomersName, City, Country
FROM Customers; -- 이렇게 줄 바꿔서 작성하는것도 가능, SQL에서는 큰 의미가 없다.
-- 대신 단어끼리는 듸어스기를 해야한다.

-- #5
SELECT 
    CustomersName, 1, 'Hello', NULL
FROM Customers
-- SELECT 뒤에 컬럼 명이 아닌 값(상수, 텍스트 등)을 직접 넣으면
-- 1) 모든 행에 값(상수 혹은 텍스트)를 똑같이 채워넣음
-- 2) 데이터가 곧 컬럼명이됨. 별도의 별칭(AS)을 지정하지 않으면 입력한 값이 그대로 해당 열의 헤더(컬럼명)이 됨

-- #6 WHERE(행 조건)
SELECT * -- 나는 가져오겠다.(SELECT) 모든 컬럼을(*)
FROM Orders -- Orders에서(FROM)
WHERE EmployeeID = 3; -- EmployeeID가 3인 조건을 충족하는 데이터를!
-- WHERE 조건

-- #9 ORDER BY
-- ASC(센딩) : 오름차순, 기본 설정이 되어있음
-- DESC(디센딩) : 내림차순, 기본 설정 값이 아님
SELECT * FROM Customers
ORDER BY ContactName; -- 기본 ASC로 지정됨
ORDER BY ContactName DESC; -- 연순을 원한다면 이렇게 진행하면 됨
-- 결과를 ContactName 기준으로 보면 알파벳 순서로 정순과 역순이 진행되는 것을 볼 수 있음

--#10 ASC와 DESC를 같이 사용할 경우
SELECT * FROM OrderDtails
ORDER BY ProductID ASC, Quantity DESC;
-- ProductID를 먼저 오름차순으로 진행하고
-- 먼저 정렬된 ProductID 값이 같은 것들 내에서 Quantity 값은 역순으로 순서를 변경하여 정리한다.

-- # 11 LIMIT
-- LIMIT {가져올 갯수} 또는 LIMIT {건널 뛸 갯수}, {가져올 갯수}
-- 가져올 갯수만 작성할 경우 건널 뛸 갯수의 기본 값은 0
-- 이런 식으로 원하는 만큼만 데이터를 가져올 수 있음
SELECT * FROM OrderDtails
LIMIT 10; -- 10개로 제한해서 가져와라
-- 순서대로 10개만 가져옴
SELECT * FROM Customers
LIMIT 0, 10 -- LIMIT 10과 결과가 같음. 건너 뛸 갯수의 기본값은 0 인가봄
LIMIT 30, 10 -- 이 경우 ID 기준으로 31~40만 나옴

-- #12 AS // 원하는 별명(alias)으로 가져오기
-- 설명 : AS를 사용하여 컬럼 며을 변경할 수 있다.
SELECT 
    CustomerId AS ID,
    CustomersName AS '고객명'
FROM Customers
