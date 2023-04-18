"""Invoke tasks for the COBRA project."""

from invoke import Collection

from . import notebook
from . import quality
# from . import docs
from . import test
# from . import ado
# from . import cruft
# from . import search
# from . import performance
# from . import export


ns = Collection()
ns.add_collection(quality, name="qa")
# ns.add_collection(docs)
ns.add_collection(test)
# ns.add_collection(ado)
# ns.add_collection(cruft)
# ns.add_collection(search)
# ns.add_collection(performance)
# ns.add_collection(export)


ns.add_collection(notebook, name="nb")
