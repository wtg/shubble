import pytest

# async test with client from conftest.py
# away makes the request to /test and checks if the response is correct
@pytest.mark.asyncio
async def test_get_test_route(client):
    response = await client.get("/test")
    assert response.status_code == 200
    assert response.json() == {"message": "ok"}