import unittest
from unittest.mock import patch

# This patch needs to be active before any google cloud libraries are imported.
patcher = patch('google.auth.default', return_value=(None, 'test-project'))
patcher.start()

import json
from unittest.mock import MagicMock
from kaybee_agent.schemas import Entity, Relationship, RelationshipIdentifier
from kaybee_agent.subagents.knowledge_graph_agent.tools import (
    upsert_entities,
    remove_entities,
    remove_relationships,
    get_relevant_neighborhoods,
)


class TestNewKnowledgeGraphTools(unittest.TestCase):

    def setUp(self):
        self.mock_tool_context = MagicMock()
        self.mock_tool_context._invocation_context.user_id = "test_user"
        self.mock_graph = {
            'entities': {
                '123': {'entity_id': '123', 'entity_names': ['Apple', 'AAPL'], 'properties': {}},
                '456': {'entity_id': '456', 'entity_names': ['Google', 'GOOG'], 'properties': {}},
            },
            'relationships': [
                {'source_entity_id': '123', 'target_entity_id': '456', 'relationship': 'is a competitor of'}
            ]
        }

    @patch('kaybee_agent.subagents.knowledge_graph_agent.tools._store_knowledge_graph')
    @patch('kaybee_agent.subagents.knowledge_graph_agent.tools._fetch_knowledge_graph')
    def test_upsert_entities_create_new(self, mock_fetch, mock_store):
        mock_fetch.return_value = self.mock_graph
        new_entities = [
            Entity(entity_names=['Microsoft', 'MSFT'], properties={'ceo': 'Satya Nadella'})
        ]
        result = upsert_entities(new_entities, self.mock_tool_context)
        self.assertEqual(result, "Entities upserted successfully.")
        mock_store.assert_called_once()
        updated_graph = mock_store.call_args[0][0]
        self.assertEqual(len(updated_graph['entities']), 3)
        # Find the new entity
        new_entity_id = None
        for entity_id, entity_data in updated_graph['entities'].items():
            if entity_data['entity_names'][0] == 'Microsoft':
                new_entity_id = entity_id
                break
        self.assertIsNotNone(new_entity_id)
        self.assertEqual(updated_graph['entities'][new_entity_id]['properties']['ceo'], 'Satya Nadella')

    @patch('kaybee_agent.subagents.knowledge_graph_agent.tools._store_knowledge_graph')
    @patch('kaybee_agent.subagents.knowledge_graph_agent.tools._fetch_knowledge_graph')
    def test_upsert_entities_update_existing(self, mock_fetch, mock_store):
        mock_fetch.return_value = self.mock_graph
        updates = [
            Entity(entity_id='123', entity_names=['Apple', 'Apple Inc.'], properties={'founder': 'Steve Jobs'})
        ]
        result = upsert_entities(updates, self.mock_tool_context)
        self.assertEqual(result, "Entities upserted successfully.")
        mock_store.assert_called_once()
        updated_graph = mock_store.call_args[0][0]
        self.assertIn('Apple Inc.', updated_graph['entities']['123']['entity_names'])
        self.assertEqual(updated_graph['entities']['123']['properties']['founder'], 'Steve Jobs')

    @patch('kaybee_agent.subagents.knowledge_graph_agent.tools._store_knowledge_graph')
    @patch('kaybee_agent.subagents.knowledge_graph_agent.tools._fetch_knowledge_graph')
    def test_upsert_entities_add_relationship(self, mock_fetch, mock_store):
        mock_fetch.return_value = self.mock_graph
        updates = [
            Entity(
                entity_id='123',
                entity_names=['Apple'],
                relationships=[Relationship(target_entity_name='Google', relationship='is a rival of')]
            )
        ]
        result = upsert_entities(updates, self.mock_tool_context)
        self.assertEqual(result, "Entities upserted successfully.")
        updated_graph = mock_store.call_args[0][0]
        self.assertEqual(len(updated_graph['relationships']), 2)


    @patch('kaybee_agent.subagents.knowledge_graph_agent.tools._store_knowledge_graph')
    @patch('kaybee_agent.subagents.knowledge_graph_agent.tools._fetch_knowledge_graph')
    def test_remove_entities(self, mock_fetch, mock_store):
        mock_fetch.return_value = self.mock_graph
        result = remove_entities(['Apple'], self.mock_tool_context)
        self.assertEqual(result, "Entities and their relationships removed successfully.")
        updated_graph = mock_store.call_args[0][0]
        self.assertNotIn('123', updated_graph['entities'])
        self.assertEqual(len(updated_graph['relationships']), 0)

    @patch('kaybee_agent.subagents.knowledge_graph_agent.tools._store_knowledge_graph')
    @patch('kaybee_agent.subagents.knowledge_graph_agent.tools._fetch_knowledge_graph')
    def test_remove_relationships(self, mock_fetch, mock_store):
        mock_fetch.return_value = self.mock_graph
        rels_to_remove = [
            RelationshipIdentifier(source_entity_name='Apple', target_entity_name='Google', relationship='is a competitor of')
        ]
        result = remove_relationships(rels_to_remove, self.mock_tool_context)
        self.assertEqual(result, "Relationships removed successfully.")
        updated_graph = mock_store.call_args[0][0]
        self.assertEqual(len(updated_graph['relationships']), 0)

    @patch('kaybee_agent.subagents.knowledge_graph_agent.tools._fetch_knowledge_graph')
    def test_get_relevant_neighborhoods_found(self, mock_fetch):
        mock_fetch.return_value = self.mock_graph
        result_str = get_relevant_neighborhoods(['Apple', 'Google'], self.mock_tool_context)
        result = json.loads(result_str)
        self.assertIn('entities', result)
        self.assertIn('relationships', result)
        self.assertIn('123', result['entities'])
        self.assertIn('456', result['entities'])
        self.assertEqual(len(result['relationships']), 1)
        self.assertEqual(result['relationships'][0]['source_entity_id'], '123')

    @classmethod
    def tearDownClass(cls):
        patcher.stop()

if __name__ == '__main__':
    unittest.main()
