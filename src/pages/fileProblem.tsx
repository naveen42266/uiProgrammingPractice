import { useState } from "react";

const file = [
    {
        name: "1",
        checked: false,
        list: [{
            name: "1:1",
            checked: false,
            list: [
                {
                    name: "1:1:1",
                    checked: false,
                    list: [
                    ]
                },
                {
                    name: "1:1:2",
                    checked: false,
                    list: [
                    ]
                }
            ]
        },
        {
            name: "1:2",
            checked: false,
            list: [
                {
                    name: "1:2:1",
                    checked: false,
                    list: [
                    ]
                },
                {
                    name: "1:2:2",
                    checked: false,
                    list: [
                    ]
                }
            ]
        }
        ]
    },
    {
        name: "2",
        checked: false,
        list: [{
            name: "2:1",
            checked: false,
            list: [
                {
                    name: "2:1:1",
                    checked: false,
                    list: [{
                        name: "2:1:1:1",
                        checked: false,
                        list: [
                        ]
                    }],
                }
            ]
        }],
    }
];

const FileProblem = () => {

    const [list, setList] = useState(file);

    const handleSelect = (itemName: string, checked: boolean) => {
        let newList = [...list]
        newList = checkList(newList, itemName, checked);
        setList(newList);
    }


    // const checkList = (newList: any, itemName: any, checked: boolean): any => {
    //     if (checked == false) {
    //         newList?.length >= 1 && newList?.map((pre: any) => {
    //             if (itemName == pre.name) {
    //                 pre.checked = !pre.checked;
    //             }
    //             else {
    //                 checkList(pre?.list, itemName, checked);
    //                 if (itemName?.length >= pre?.name?.length) {
    //                     const first = itemName.slice(0, pre.name.length);
    //                     if (first == pre.name) {
    //                         pre.checked = !pre.checked;
    //                     }
    //                 }
    //             }
    //         })
    //     }
    //     else {
    //         newList?.length >= 1 && newList?.map((pre: any) => {
    //             if (itemName == pre.name) {
    //                 pre.checked = false;
    //             }
    //             else {
    //                 checkList(pre?.list, itemName, checked);
    //             }
    //         })
    //     }
    //     return newList;
    // }


    // console.log(list?.find((item) => item.name === "1")?.list?.find((item) => item.name === "1:1")?.list?.find((item) => item.name === "1:1:2")?.checked);

    
// const checkList = (
//     newList: any[],
//     itemName: string,
//     checked: boolean
// ): any[] => {

//     newList?.forEach((item: any) => {

//         // Exact item
//         if (item.name === itemName) {
//             item.checked = !checked;
//         }

//         // Parent of selected item
//         else if (
//             itemName.startsWith(item.name + ":")
//         ) {
//             item.checked = !checked;
//         }

//         // Continue searching children
//         if (item.list?.length) {
//             checkList(item.list, itemName, checked);
//         }
//     });

//     return newList;
// };


const checkList = (
    list: any[],
    itemName: string,
    checked: boolean
): any[] => {

    list.forEach((item) => {

        // Selected item
        if (item.name === itemName) {
            item.checked = !checked;

            // If unchecking → uncheck all children
            if (checked) {
                uncheckChildren(item.list);
            }

            return;
        }

        // Parent of selected item
        if (itemName.startsWith(item.name + ":")) {

            if (!checked) {
                // Selecting → parent checked
                item.checked = true;
            }

            // Unselecting → parent should be checked
            // only if another child is still checked
            else {
                item.checked = item.list?.some(
                    (child: any) => child.checked
                ) ?? false;
            }
        }

        // Continue recursively
        if (item.list?.length) {
            checkList(item.list, itemName, checked);
        }
    });

    return list;
};


const uncheckChildren = (list: any[]) => {
    list?.forEach((item) => {
        item.checked = false;

        if (item.list?.length) {
            uncheckChildren(item.list);
        }
    });
};

    const InputField = (item: any) => {


        return (
            <div className="flex justify-start gap-4 items-start">
                <input type="checkbox" checked={item.checked} className="mt-1.5" onChange={() => { handleSelect(item.name, item.checked) }} />
                <div>{item.name}</div>
                <div className="">
                    {item?.list?.map((subItem: any, index: number) => {
                        return (
                            <div key={index}>
                                {InputField(subItem)}
                            </div>
                        )
                    })}
                </div>
            </div>
        )
    }

    return (
        <div>
            <h1>File Problem</h1>
            <p>This is a simple file problem component.</p>
            {list?.map((item, index) => {
                return (
                    <div key={index}>
                        {InputField(item)}
                    </div>
                )
            })}


        </div>
    )
};
export default FileProblem;