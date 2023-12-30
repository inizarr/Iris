import streamlit as st
from graphviz import Digraph

def decision_tree_visualization():
    st.write('Menggunakan Graphviz untuk visualisasi pohon keputusan.')

    # Buat pohon keputusan di sini.
    tree = Digraph('tree', node_attr={'shape': 'box'})
    tree.edge('Parent', 'Child')
    # Tambahkan pohon keputusan lainnya di sini...

    # Render pohon keputusan menggunakan Graphviz
    st.graphviz_chart(tree)

decision_tree_visualization()
