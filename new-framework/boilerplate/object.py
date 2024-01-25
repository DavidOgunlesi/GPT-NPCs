from component import ScriptComponent


class Object:

  def __init__(self):
    self.components = []

  def AddComponent(self, components: ScriptComponent):
    self.components.append(components)
  
  def Start(self):
    for component in self.components:
      component.Start()

  def Update(self):
    for component in self.components:
      component.Update()