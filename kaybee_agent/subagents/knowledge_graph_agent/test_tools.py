import json
import unittest
from unittest.mock import patch, MagicMock

from .tools import (
    get_entity_neighborhood,
    add_synonyms,
    add_relationship,
    remove_synonyms,
    delete_entity,
    remove_relationship
)


class TestKnowledgeGraphTools(unittest.TestCase):

    def setUp(self):
        self.mock_graph = {
            'entities': {
                '123': {'entity_id': '123', 'entity_names': ['Apple', 'AAPL']},
                '456': {'entity_id': '456', 'entity_names': ['Google', 'GOOG']},
            },
            'relationships': [
                {'source_entity_id': '123', 'target_entity_id': '456', 'relationship': 'is a competitor of'}
            ]
        }

    @patch('kaybee_agent.subagents.knowledge_graph_agent.tools.store_knowledge_graph')
    @patch('kaybee_agent.subagents.knowledge_graph_agent.tools.fetch_knowledge_graph')
    def test_add_synonyms(self, mock_fetch, mock_store):
        mock_fetch.return_value = self.mock_graph
        result = add_synonyms('123', ['Apple Inc.'])
        self.assertIn("Synonyms added successfully", result)
        mock_store.assert_called_once()
        updated_graph = mock_store.call_args[0][0]
        self.assertIn('Apple Inc.', updated_graph['entities']['123']['entity_names'])

    @patch('kaybee_agent.subagents.knowledge_graph_agent.tools.store_knowledge_graph')
    @patch('kaybee_agent.subagents.knowledge_graph_agent.tools.fetch_knowledge_graph')
    def test_remove_synonyms(self, mock_fetch, mock_store):
        mock_fetch.return_value = self.mock_graph
        result = remove_synonyms('123', ['AAPL'])
        self.assertIn("Synonyms removed successfully", result)
        mock_store.assert_called_once()
        updated_graph = mock_store.call_args[0][0]
        self.assertNotIn('AAPL', updated_graph['entities']['123']['entity_names'])

    @patch('kaybee_agent.subagents.knowledge_graph_agent.tools.store_knowledge_graph')
    @patch('kaybee_agent.subagents.knowledge_graph_agent.tools.fetch_knowledge_graph')
    def test_add_relationship(self, mock_fetch, mock_store):
        mock_fetch.return_value = self.mock_graph
        result = add_relationship('123', 'is based in', '456')
        self.assertIn("Relationship 'Apple -> is based in -> Google' added successfully", result)
        mock_store.assert_called_once()
        updated_graph = mock_store.call_args[0][0]
        self.assertIn({'source_entity_id': '123', 'target_entity_id': '456', 'relationship': 'is based in'}, updated_graph['relationships'])

    @patch('kaybee_agent.subagents.knowledge_graph_agent.tools.store_knowledge_graph')
    @patch('kaybee_agent.subagents.knowledge_graph_agent.tools.fetch_knowledge_graph')
    def test_remove_relationship(self, mock_fetch, mock_store):
        mock_fetch.return_value = self.mock_graph
        result = remove_relationship('123', 'is a competitor of', '456')
        self.assertIn("Relationship 'Apple -> is a competitor of -> Google' removed successfully", result)
        mock_store.assert_called_once()
        updated_graph = mock_store.call_args[0][0]
        self.assertEqual(len(updated_graph['relationships']), 0)

    @patch('kaybee_agent.subagents.knowledge_graph_agent.tools.store_knowledge_graph')
    @patch('kaybee_agent.subagents.knowledge_graph_agent.tools.fetch_knowledge_graph')
    def test_delete_entity(self, mock_fetch, mock_store):
        mock_fetch.return_value = self.mock_graph
        result = delete_entity('123')
        self.assertIn("Entity 'Apple' and its relationships deleted successfully", result)
        mock_store.assert_called_once()
        updated_graph = mock_store.call_args[0][0]
        self.assertNotIn('123', updated_graph['entities'])
        self.assertEqual(len(updated_graph['relationships']), 0)

    @patch('kaybee_agent.subagents.knowledge_graph_agent.tools.fetch_knowledge_graph')
    def test_get_entity_neighborhood_found(self, mock_fetch):
        mock_fetch.return_value = self.mock_graph
        result_str = get_entity_neighborhood('Apple')
        result = json.loads(result_str)
        self.assertIn('entities', result)
        self.assertIn('relationships', result)
        self.assertIn('123', result['entities'])
        self.assertIn('456', result['entities'])
        self.assertEqual(len(result['relationships']), 1)
        self.assertEqual(result['relationships'][0]['source_entity_id'], '123')

    @patch('kaybee_agent.subagents.knowledge_graph_agent.tools.fetch_knowledge_graph')
    def test_get_entity_neighborhood_not_found(self, mock_fetch):
        mock_fetch.return_value = self.mock_graph
        result = get_entity_neighborhood('Microsoft')
        self.assertEqual(result, "Error: Entity 'Microsoft' not found.")

if __name__ == '__main__':
    unittest.main()
