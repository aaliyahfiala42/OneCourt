import argparse
import json

from collections import defaultdict

import numpy as np


class Node(object):
    # pylint: disable=too-many-instance-attributes, too-few-public-methods
    def __init__(self, frame_idx, index, uv, score, stay_node):
        self.frame_idx = frame_idx
        self.index = index
        self.uv = uv
        self.score = score
        self.incoming_nodes = []
        self.stay_node = stay_node
        self.shortest_path = None
        self.shortest_path_score = 0

    def add_edge(self, node, max_distance=None):
        if max_distance is None:
            self.incoming_nodes.append(node.index)
        else:
            distance = np.linalg.norm(np.array(self.uv) - np.array(node.uv))

            if distance <= max_distance:
                self.incoming_nodes.append(node.index)

def dynamic_path_shortest(node, all_nodes):
    if node.shortest_path is not None:
        return node.shortest_path, node.shortest_path_score

    max_score = 0
    best_shortest_path = []

    node_score = node.score
    stay_node = node.stay_node

    incoming_nodes = node.incoming_nodes

    for node_index in incoming_nodes:
        cur_node = all_nodes.get(node_index)

        cur_shortest_path, cur_score = dynamic_path_shortest(cur_node, all_nodes)

        candidate_score = cur_score + node_score
        if candidate_score > max_score:
            max_score = candidate_score
            best_shortest_path = cur_shortest_path + [node.index]

    node.shortest_path = best_shortest_path
    node.shortest_path_score = max_score

    return best_shortest_path, max_score

# def run_ball_tracking_dp(log, detections, params, parabola_interpolation_params):
#     detections = filter_non_likely_balls(detections, params.likely_score_cutoff)
#     output = []

#     prev_track = None
#     for track_id in range(params.num_tracks):
#         log.info("Finding track {}".format(track_id))

#         detections = remove_prev_track(detections, prev_track)
#         parabola_detections, track = build_and_solve_dp(log, detections, params,
#                                                         parabola_interpolation_params)

#         if parabola_detections:
#             log.info("Rerunning ball tracking dp with {} additional detections".format(
#                 len(parabola_detections)))

#             detections_with_parabolas = detections + parabola_detections
#             _, track = build_and_solve_dp(log, detections_with_parabolas, params)

#         for position_dict in track:
#             position_dict["trackId"] = track_id

#         prev_track = track
#         output += track

#     return output


def main(args):
    with open(args.detections) as f:
        detections = json.load(f)

    output = build_and_solve_dp(detections,
        args.distance_cutoff, args.max_entry_frame, args.stay_score)

    with open(args.output_file, "w") as f:
        json.dump(output, f, indent=4)

def build_and_solve_dp(detections, distance_cutoff, max_entry_frame, stay_score):
    nodes_dict = defaultdict(list)
    all_nodes = {}

    detections = sorted(detections, key=lambda x: x["frameIdx"])
    num_frames = max([x["frameIdx"] for x in detections])

    max_node_index = 0
    for detection in detections:
        cur_node_index = max_node_index
        frame_idx = detection["frameIdx"]
        uv = detection["uv"]
        score = detection["score"]

        node = Node(frame_idx, cur_node_index, uv, score, False)
        nodes_dict[frame_idx].append(node)
        all_nodes[cur_node_index] = node
        max_node_index += 1

        # Duplicate the detection till the last frame and add edge
        last_node = node
        for stay_frame_idx in range(frame_idx + 1, num_frames):
            node = Node(stay_frame_idx, max_node_index, uv, stay_score, True)

            node.add_edge(last_node)
            nodes_dict[stay_frame_idx].append(node)
            all_nodes[max_node_index] = node
            last_node = node
            max_node_index += 1

    # Add source and sink node
    source_node = Node(-1, -1, None, None, False)
    all_nodes[-1] = source_node

    sink_node = Node(num_frames, max_node_index, None, 0, False)
    all_nodes[max_node_index] = sink_node

    # Set shortest path from source to source
    source_node.shortest_path = [-1]

    # Add edges between all real nodes (except source and sink)
    for frame_idx in range(1, num_frames):
        cur_frame_nodes = nodes_dict.get(frame_idx, [])
        last_frame_nodes = nodes_dict.get(frame_idx - 1, [])

        for cur_node in cur_frame_nodes:
            max_distance = distance_cutoff
            if cur_node.stay_node:
                continue

            for last_node in last_frame_nodes:
                cur_node.add_edge(last_node, max_distance)

    # Add edges from source node and to sink node
    for frame_idx, frame_nodes in nodes_dict.items():
        if frame_idx <= max_entry_frame:
            for node in frame_nodes:
                node.add_edge(source_node)

        if frame_idx == num_frames - 1:
            for node in frame_nodes:
                sink_node.add_edge(node)

    shortest_path_nodes, shortest_path_score = dynamic_path_shortest(sink_node, all_nodes)

    print("Obtained track with score: {}".format(shortest_path_score))

    keyed_ball_track = {}
    for node_index in shortest_path_nodes:
        if node_index == -1 or node_index == max_node_index:
            continue

        node = all_nodes.get(node_index)
        frame_idx = node.frame_idx
        node_uv = node.uv

        if node.stay_node:
            continue

        keyed_ball_track[frame_idx] = node_uv

    # If no track could be obtained, return empty output
    if not keyed_ball_track:
        return [], []


    # Todo 
    # interpolate_stay_nodes(log, keyed_ball_track, num_frames)

    output = []
    for frame_idx, uv in keyed_ball_track.items():
        output.append({
            "frameIdx": frame_idx,
            "uv": uv,
            "score": shortest_path_score,
        })

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('detections', type=str, help="Input detections JSON file")
    parser.add_argument('output_file', type=str, help="Output track json")
    parser.add_argument("--distance_cutoff", type=float, default=30, help="Distance in px that ball can move in one frame")
    parser.add_argument("--max_entry_frame", type=int, default=80, help="Max frame the ball can enter")
    parser.add_argument("--stay_score", type=float, default=0.1, help="Score of a stay node")
    main(parser.parse_args())