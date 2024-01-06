// use crate::components::{
//     graph::Graph,
//     node::{AccFn, HiddenNode, InputNode, NodeType, OutputNode},
//     utils::Activations, edge::Linear,
// };

// pub struct PiNNConstructor {
//     vg_subnet_width: Vec<usize>,
//     vd_subnet_width: Vec<usize>,
// }

// impl PiNNConstructor {
//     pub fn new(vg_subnet_width: Vec<usize>, vd_subet_width: Vec<usize>) -> Self {
//         assert_eq!(vg_subnet_width.len(), vd_subet_width.len());
//         Self {
//             vg_subnet_width: vg_subnet_width,
//             vd_subnet_width: vd_subet_width,
//         }
//     }

//     pub fn construct(&self) -> Graph {
//         let mut pinn = Graph::new();

//         let input_vd = NodeType::Input(InputNode::new(
//             1,
//             Activations::Id,
//             AccFn::Sum,
//             "vd_input",
//             &["V(b_ds)"],
//         ));
//         let input_vg = NodeType::Input(InputNode::new(
//             1,
//             Activations::Id,
//             AccFn::Sum,
//             "vg_input",
//             &["V(b_gs)"],
//         ));
//         let vd_subs = self.vd_subnet_width.iter().map(|x| {
//             NodeType::Hidden(HiddenNode::new(
//                 *x,
//                 Activations::Sigmoid,
//                 AccFn::Sum,
//                 Box::leak(format!("vd_sub{}", x).into_boxed_str()),
//             ))
//         });
//         let vg_subs = self.vg_subnet_width.iter().map(|x| {
//             NodeType::Hidden(HiddenNode::new(
//                 *x,
//                 Activations::Tanh,
//                 AccFn::Sum,
//                 Box::leak(format!("vg_sub{}", x).into_boxed_str()),
//             ))
//         });
//         let output = NodeType::Output(OutputNode::new(
//             1,
//             Activations::Id,
//             AccFn::Prod,
//             "ids_output",
//             &["I(b_ds)"],
//         ));

//         for i in 0..vd_subs.len() {

//         }

//         pinn
//     }
// }

#[test]
fn test_pinn_construct() {
    use crate::components::{
        edge::Linear,
        graph::Graph,
        node::{AccFn, HiddenNode, InputNode, NodeType, OutputNode},
        utils::Activations,
    };
    use tch::nn::{self, LinearConfig};

    let vs = nn::VarStore::new(tch::Device::Cpu);
    let mut pinn = Graph::new();
    let input_vd = NodeType::Input(InputNode::new(
        1,
        Activations::Id,
        AccFn::Sum,
        "vd_input",
        &["V(b_ds)"],
    ));
    let input_vg = NodeType::Input(InputNode::new(
        1,
        Activations::Id,
        AccFn::Sum,
        "vg_input",
        &["V(b_gs)"],
    ));
    let vd_sub1 = NodeType::Hidden(HiddenNode::new(
        20,
        Activations::Tanh,
        AccFn::Sum,
        "vd_sub1",
    ));
    let vd_sub2 = NodeType::Hidden(HiddenNode::new(1, Activations::Tanh, AccFn::Sum, "vd_sub2"));
    let vg_sub1 = NodeType::Hidden(HiddenNode::new(
        30,
        Activations::Sigmoid,
        AccFn::Sum,
        "vg_sub1",
    ));
    let vg_sub2 = NodeType::Hidden(HiddenNode::new(
        1,
        Activations::Sigmoid,
        AccFn::Sum,
        "vg_sub2",
    ));
    let output = NodeType::Output(OutputNode::new(
        1,
        Activations::Id,
        AccFn::Prod,
        "ids_output",
        &["I(b_ds)"],
    ));
    let id2vd1 = nn::linear(
        vs.root(),
        1,
        20,
        LinearConfig {
            ws_init: nn::init::DEFAULT_KAIMING_UNIFORM,
            bs_init: None,
            bias: false,
        },
    );
    let vd12vd2 = nn::linear(
        vs.root(),
        20,
        1,
        LinearConfig {
            ws_init: nn::init::DEFAULT_KAIMING_UNIFORM,
            bs_init: None,
            bias: false,
        },
    );
    let vd22o = nn::linear(
        vs.root(),
        1,
        1,
        LinearConfig {
            ws_init: nn::init::DEFAULT_KAIMING_UNIFORM,
            bs_init: None,
            bias: false,
        },
    );
    let ig2vg1 = nn::linear(
        vs.root(),
        1,
        30,
        LinearConfig {
            ws_init: nn::init::DEFAULT_KAIMING_UNIFORM,
            bs_init: None,
            bias: false,
        },
    );
    let vg12vg2 = nn::linear(
        vs.root(),
        30,
        1,
        LinearConfig {
            ws_init: nn::init::DEFAULT_KAIMING_UNIFORM,
            bs_init: None,
            bias: false,
        },
    );
    let vg22o = nn::linear(
        vs.root(),
        1,
        1,
        LinearConfig {
            ws_init: nn::init::DEFAULT_KAIMING_UNIFORM,
            bs_init: None,
            bias: false,
        },
    );
    let vd12vg1 = nn::linear(
        vs.root(),
        20,
        30,
        LinearConfig {
            ws_init: nn::init::DEFAULT_KAIMING_UNIFORM,
            bs_init: None,
            bias: false,
        },
    );
    let vd22vg2 = nn::linear(
        vs.root(),
        1,
        1,
        LinearConfig {
            ws_init: nn::init::DEFAULT_KAIMING_UNIFORM,
            bs_init: None,
            bias: false,
        },
    );

    pinn.add_edge(Linear::new(input_vd, vd_sub1, id2vd1));
    pinn.add_edge(Linear::new(input_vg, vg_sub1, ig2vg1));
    pinn.add_edge(Linear::new(vd_sub1, vg_sub1, vd12vg1));
    pinn.add_edge(Linear::new(vd_sub1, vd_sub2, vd12vd2));
    pinn.add_edge(Linear::new(vg_sub1, vg_sub2, vg12vg2));
    pinn.add_edge(Linear::new(vd_sub2, vg_sub2, vd22vg2));
    pinn.add_edge(Linear::new(vd_sub2, output, vd22o));
    pinn.add_edge(Linear::new(vg_sub2, output, vg22o));
}
