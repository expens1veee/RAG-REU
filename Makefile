TARGET_DIR=.

PYLINT_CONFIG=../.pylintrc

clean:
	rm -rf logs/*

lint:
	pylint --rcfile=$(PYLINT_CONFIG) $(TARGET_DIR)

deps:
	docker pull qdrant/qdrant
	pip install -r requirements

run:
	python main.py

qdrant.start:
	docker run -d -p 6333:6333 -p 6334:6334 --name qdrant qdrant/qdrant

qdrant.stop:
	docker stop qdrant

venv:
	source .venv/bin/activate