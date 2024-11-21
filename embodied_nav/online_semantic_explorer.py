import airsim
from .airsim_explorer import AirSimExplorer
from .spatial_relationship_extractor import SpatialRelationshipExtractor
from .llm import LLMInterface
import networkx as nx
import logging
import asyncio
import time
import os
import signal
from datetime import datetime
import threading
from .config import Config

class OnlineSemanticExplorer(AirSimExplorer):
    def __init__(self):
        super().__init__()
        self.llm_interface = LLMInterface()
        self.relationship_extractor = SpatialRelationshipExtractor(self.llm_interface)
        self.last_forest_update = time.time()
        self.forest_update_interval = Config.ONLINE_SEMANTIC['forest_update_interval']
        self.semantic_forest = nx.Graph()
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        # Ensure ENVIRONMENT_NAME is set
        if not hasattr(self, 'ENVIRONMENT_NAME'):
            self.ENVIRONMENT_NAME = 'Building99'
        
        # Add signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print("\nReceived shutdown signal. Building final semantic forest...")
        try:
            # Run one final forest update synchronously
            asyncio.run_coroutine_threadsafe(self.final_update(), self.loop).result()
            print("Final semantic forest update complete")
        except Exception as e:
            print(f"Error during final update: {str(e)}")
        finally:
            self.shutdown()
            os._exit(0)

    async def final_update(self):
        """Ensure final semantic forest update completes"""
        try:
            print("\nStarting final semantic forest update...")
            await self.update_semantic_forest()
            self.save_semantic_forest(is_final=True)  # Save as final version
            print("Final semantic forest saved successfully")
        except Exception as e:
            print(f"Error during final semantic forest update: {str(e)}")

    async def update_semantic_forest(self):
        """Update semantic forest with current objects"""
        try:
            print("\n=== Updating Semantic Forest ===")
            
            # Get all non-drone objects from current graph
            objects = [
                {'id': node, **{k:v for k,v in data.items() if k != 'level'}}
                for node, data in self.G.nodes(data=True)
                if data.get('type') != 'drone'
            ]
            
            # Check minimum objects requirement
            if len(objects) < Config.ONLINE_SEMANTIC['min_objects_for_update']:
                print(f"Not enough objects to process yet (minimum {Config.ONLINE_SEMANTIC['min_objects_for_update']} required)")
                return
                
            print(f"Processing {len(objects)} objects for semantic relationships")
            
            # Extract relationships and create hierarchical clusters
            self.semantic_forest = await self.relationship_extractor.extract_relationships(objects)
            print(f"Updated semantic forest with {len(self.semantic_forest.nodes)} nodes")
        except Exception as e:
            print(f"Error updating semantic forest: {str(e)}")
            import traceback
            print(traceback.format_exc())

    def save_semantic_forest(self, is_final=False):
        """Save the current state of the semantic forest
        Args:
            is_final (bool): If True, save as final version. If False, save as ongoing version.
        """
        try:
            if not hasattr(self, 'semantic_forest') or len(self.semantic_forest.nodes) == 0:
                print("\nNo nodes in semantic forest to save")
                return

            env_name = getattr(self, 'ENVIRONMENT_NAME', 'Building99')
            
            # Choose filename based on whether this is the final save or ongoing
            if is_final:
                filename = f"final_online_semantic_forest_{env_name}.gml"
            else:
                filename = f"online_semantic_forest_{env_name}.gml"
                
            filepath = os.path.join(self.output_dir, filename)
            
            # Save the semantic forest
            nx.write_gml(self.semantic_forest, filepath)
            print(f"\nSaved semantic forest with {len(self.semantic_forest.nodes)} nodes and {len(self.semantic_forest.edges)} edges")
            print(f"Forest saved to: {filepath}")
            
        except Exception as e:
            print(f"\nError saving semantic forest: {str(e)}")
            import traceback
            print(traceback.format_exc())

    async def run_async(self):
        """Async run method to handle periodic updates"""
        while self.is_running:
            current_time = time.time()
            if current_time - self.last_forest_update >= self.forest_update_interval:
                await self.update_semantic_forest()
                self.save_semantic_forest(is_final=False)  # Save as ongoing version
                self.last_forest_update = current_time
            await asyncio.sleep(1)

    def run(self):
        """Override run to include async forest updates"""
        try:
            # Start async update task
            self.loop.create_task(self.run_async())
            
            # Start the event loop
            threading.Thread(target=self.loop.run_forever, daemon=True).start()
            
            # Call parent run method
            super().run()
            
        except Exception as e:
            logging.error(f"Error during run: {e}")
            self.shutdown()

    def shutdown(self):
        """Override shutdown to ensure forest is completed"""
        print("\nInitiating shutdown sequence...")
        try:
            self.is_running = False
            
            # Ensure any pending async operations complete
            if hasattr(self, 'loop') and self.loop.is_running():
                try:
                    # Run one final update if we haven't already
                    if not hasattr(self, 'final_update_complete'):
                        # Use timeout from config
                        asyncio.run_coroutine_threadsafe(
                            self.final_update(), 
                            self.loop
                        ).result(timeout=Config.ONLINE_SEMANTIC['max_update_timeout'])
                except Exception as e:
                    print(f"Error during final forest update: {str(e)}")
                finally:
                    self.loop.stop()
            
            if hasattr(self, 'loop') and not self.loop.is_closed():
                self.loop.close()
                
        except Exception as e:
            print(f"Error during shutdown: {str(e)}")
        finally:
            super().shutdown()

if __name__ == "__main__":
    explorer = None
    try:
        explorer = OnlineSemanticExplorer()
        explorer.run()
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received")
        if explorer:
            explorer.signal_handler(signal.SIGINT, None)
    except Exception as e:
        print(f"Error: {str(e)}")
        if explorer:
            explorer.shutdown()
    finally:
        if explorer:
            print("\nEnsuring final cleanup...")
            # One last attempt to save if everything else failed
            try:
                asyncio.run(explorer.final_update())
            except Exception as e:
                print(f"Final cleanup attempt failed: {str(e)}")
        print("Cleanup complete")
        os._exit(0)