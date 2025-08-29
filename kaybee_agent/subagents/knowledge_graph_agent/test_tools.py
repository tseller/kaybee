import unittest
from unittest.mock import patch, MagicMock

# This patch needs to be active before any google cloud libraries are imported.
patcher = patch('google.auth.default', return_value=(None, 'test-project'))
patcher.start()

# Patch os.environ before importing the tools module
env_patcher = patch.dict('os.environ', {'KNOWLEDGE_GRAPH_BUCKET': 'test-bucket'})
env_patcher.start()

from kaybee_agent.subagents.knowledge_graph_agent.tools import (
    get_relevant_neighborhoods,
    store_graph,
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

    @patch('kaybee_agent.subagents.knowledge_graph_agent.tools._fetch_knowledge_graph')
    def test_get_relevant_neighborhoods_found(self, mock_fetch):
        mock_fetch.return_value = self.mock_graph
        result = get_relevant_neighborhoods(['Apple', 'Google'], self.mock_tool_context)
        self.assertIn('entities', result)
        self.assertIn('relationships', result)
        # The result from node_link_data is a list of dicts, not a dict of dicts
        entities_as_dict = {e['id']: e for e in result['entities']}
        self.assertIn('123', entities_as_dict)
        self.assertIn('456', entities_as_dict)
        self.assertEqual(len(result['relationships']), 1)
        self.assertEqual(result['relationships'][0]['source'], '123')


    @patch('kaybee_agent.subagents.knowledge_graph_agent.tools._store_knowledge_graph')
    def test_store_graph(self, mock_store):
        result = store_graph(self.mock_graph, self.mock_tool_context)
        self.assertEqual(result, "Graph stored successfully.")
        mock_store.assert_called_once_with(knowledge_graph=self.mock_graph, graph_id="test_user")


    @classmethod
    def tearDownClass(cls):
        patcher.stop()
        env_patcher.stop()

if __name__ == '__main__':
    unittest.main()
