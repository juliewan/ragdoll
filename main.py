import argparse
from ragdoll import Ragdoll

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', help='directory of *.pdfs or cloud collection name')
    parser.add_argument('-l', '--local', action='store_true', help='is local directory (optional)')
    parser.add_argument('-p', '--persist', action='store_true', help='upload as cloud collection (optional)')
    parser.add_argument('-t', '--temp', type=float, help='model temperature (optional)', default='0.25')
    parser.add_argument('-n', '--num_pred', type=int, help='max tokens to generate (optional)', default='512')
    args = parser.parse_args()

    model = Ragdoll(**vars(args))
    model.build_vector_store()
    model.build_react_graph()

    while True:
        prompt = input("\n\nEnter prompt (or 'q' to exit): ").strip().lower()
        if prompt == "q":
            break

        model.respond(prompt)