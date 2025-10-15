lint:
	ruff check .

format:
	ruff format .

up:
	docker-compose up -d

down:
	docker-compose down
