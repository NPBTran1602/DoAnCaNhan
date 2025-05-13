# DoAnCaNhan

# AI
Báo cáo Đồ án Kết thúc Học phần Trí tuệ Nhân tạo

Đề tài: Giải bài toán 8-Puzzle bằng các thuật toán tìm kiếm

Giảng viên hướng dẫn: Phan Thị Huyền Trang

Sinh viên thực hiện: Nguyễn Phạm Bảo Trân - 23110348

Ngày báo cáo: Tháng 05 năm 2025


1. Mục tiêu


Mục tiêu của đồ án này là tìm hiểu, triển khai và đánh giá hiệu quả của các thuật toán tìm kiếm khác nhau trong lĩnh vực Trí tuệ Nhân tạo để giải quyết bài toán kinh điển 8-Puzzle. Cụ thể, nhóm hướng tới:

Nắm vững lý thuyết về các thuật toán tìm kiếm có thông tin và không có thông tin, tìm kiếm cục bộ, tìm kiếm dựa trên ràng buộc, tìm kiếm trong môi trường phức tạp và học tăng cường.

Xây dựng chương trình mô phỏng bài toán 8-Puzzle với giao diện đồ họa thân thiện, cho phép người dùng nhập trạng thái ban đầu và trạng thái đích.

Tích hợp các thuật toán tìm kiếm đã triển khai vào chương trình mô phỏng.

Đánh giá hiệu suất (thời gian thực thi, số node mở rộng) của các thuật toán trên các bài toán 8-Puzzle khác nhau.

Trực quan hóa quá trình tìm kiếm và lời giải của từng thuật toán.

So sánh và đưa ra nhận xét về ưu nhược điểm của từng thuật toán trong việc giải quyết bài toán 8-Puzzle.


2. Nội dung


2.1. Bài toán tìm kiếm và các khái niệm cơ bản
   
Bài toán 8-Puzzle là một ví dụ điển hình của bài toán tìm kiếm trạng thái trong AI. Mục tiêu là di chuyển các ô vuông (được đánh số từ 1 đến 8) trên một bàn cờ 3x3 để đạt được một trạng thái đích cho trước, bằng cách trượt ô trống (số 0) vào các vị trí liền kề.

Các thành phần chính của bài toán tìm kiếm bao gồm:

Không gian trạng thái (State Space): Tập hợp tất cả các cấu hình có thể có của bàn cờ 8-Puzzle. Mỗi trạng thái là một cách sắp xếp các số từ 0 đến 8 trên bàn cờ 3x3.

Trạng thái ban đầu (Initial State): Cấu hình xuất phát của bàn cờ.

Các hành động/toán tử (Actions/Operators): Các thao tác di chuyển ô trống. Trong bài toán 8-Puzzle, các hành động có thể là: Lên (Up), Xuống (Down), Sang trái (Left), Sang phải (Right), tùy thuộc vào vị trí hiện tại của ô trống.

Mô hình chuyển đổi (Transition Model): Xác định trạng thái kết quả sau khi thực hiện một hành động từ một trạng thái nhất định.

Kiểm tra đích (Goal Test): Một hàm kiểm tra xem một trạng thái có phải là trạng thái đích hay không.

Chi phí đường đi (Path Cost): Chi phí tích lũy của một chuỗi hành động từ trạng thái ban đầu đến trạng thái hiện tại. Trong bài toán 8-Puzzle đơn giản, chi phí cho mỗi bước di chuyển thường là 1.

Lời giải (Solution): Lời giải cho bài toán tìm kiếm là một chuỗi các hành động từ trạng thái ban đầu dẫn đến trạng thái đích. Lời giải tối ưu là lời giải có chi phí đường đi thấp nhất.


2.2. Các thuật toán Tìm kiếm không có thông tin (Uninformed Search)

Các thuật toán tìm kiếm không có thông tin (hay còn gọi là Blind Search) chỉ sử dụng thông tin về trạng thái hiện tại và các hành động có thể có, mà không sử dụng bất kỳ thông tin bổ sung nào về "độ gần" của trạng thái hiện tại với trạng thái đích.

Tìm kiếm theo chiều rộng (Breadth-First Search - BFS):

Nguyên lý: Mở rộng tất cả các node ở độ sâu d trước khi chuyển sang các node ở độ sâu d+1. Sử dụng hàng đợi (queue) để lưu trữ các node cần mở rộng.

Áp dụng cho 8-Puzzle: Bắt đầu từ trạng thái ban đầu, thuật toán khám phá tất cả các trạng thái có thể đạt được sau 1 bước di chuyển, sau đó là 2 bước, v.v., cho đến khi tìm thấy trạng thái đích.

Ưu điểm: Đảm bảo tìm thấy lời giải tối ưu (với chi phí bước đi bằng 1) nếu có lời giải.

Nhược điểm: Có thể yêu cầu bộ nhớ rất lớn do phải lưu trữ tất cả các node ở mỗi cấp độ.

Hình ảnh GIF minh họa:

Nhận xét về hiệu suất trên 8-Puzzle: BFS thường hiệu quả với các bài toán có lời giải ở độ sâu nông. Tuy nhiên, với các bài toán 8-Puzzle yêu cầu nhiều bước di chuyển, không gian tìm kiếm có thể rất lớn, khiến BFS trở nên chậm và tốn bộ nhớ.

Tìm kiếm theo chiều sâu (Depth-First Search - DFS):

Nguyên lý: Luôn mở rộng node sâu nhất có thể trong cây tìm kiếm. Sử dụng ngăn xếp (stack) để lưu trữ các node cần mở rộng. Có thể cài đặt đệ quy.

Áp dụng cho 8-Puzzle: Thuật toán đi sâu vào một nhánh của cây tìm kiếm cho đến khi đạt đến độ sâu tối đa hoặc không thể mở rộng thêm, sau đó quay lui và khám phá nhánh khác.

Ưu điểm: Yêu cầu bộ nhớ ít hơn đáng kể so với BFS.

Nhược điểm: Không đảm bảo tìm thấy lời giải tối ưu và có thể rơi vào vòng lặp vô hạn nếu không có cơ chế kiểm tra trạng thái đã thăm hoặc giới hạn độ sâu.

Hình ảnh GIF minh họa:

Nhận xét về hiệu suất trên 8-Puzzle: DFS có thể tìm thấy lời giải nhanh nếu lời giải nằm ở độ sâu lớn dọc theo nhánh đầu tiên nó khám phá. Tuy nhiên, nếu không có giới hạn độ sâu, nó có thể tìm kiếm rất lâu trong các nhánh không chứa lời giải.

Tìm kiếm chi phí đồng nhất (Uniform-Cost Search - UCS):

Nguyên lý: Mở rộng node có chi phí đường đi từ gốc đến node đó (g(n)) thấp nhất. Sử dụng hàng đợi ưu tiên (priority queue) sắp xếp theo chi phí g(n).

Áp dụng cho 8-Puzzle: Tìm kiếm đường đi có tổng chi phí thấp nhất. Với chi phí mỗi bước là 1, UCS tương tự như BFS.

Ưu điểm: Đảm bảo tìm thấy lời giải tối ưu.

Nhược điểm: Giống BFS, có thể tốn nhiều bộ nhớ và thời gian với không gian trạng thái lớn.

Hình ảnh GIF minh họa: (Cần bổ sung hình ảnh GIF quá trình UCS trên 8-Puzzle)

Hình ảnh so sánh hiệu suất: (Cần bổ sung biểu đồ so sánh hiệu suất UCS với các thuật toán khác)

Nhận xét về hiệu suất trên 8-Puzzle: Tương tự BFS khi chi phí bước đi là 1.

Tìm kiếm theo chiều sâu lặp sâu dần (Iterative Deepening Depth-First Search - IDDFS):

Nguyên lý: Kết hợp ưu điểm của DFS (bộ nhớ ít) và BFS (tìm thấy lời giải tối ưu). Thực hiện chuỗi các lần tìm kiếm theo chiều sâu với giới hạn độ sâu tăng dần (1, 2, 3, ...).

Áp dụng cho 8-Puzzle: Tìm kiếm đến độ sâu 1, nếu không tìm thấy thì tìm kiếm đến độ sâu 2, v.v.

Ưu điểm: Tìm thấy lời giải tối ưu và yêu cầu bộ nhớ ít như DFS. Thường là thuật toán tìm kiếm không có thông tin tốt nhất cho các bài toán có không gian trạng thái lớn và độ sâu lời giải không quá lớn.

Nhược điểm: Lặp lại việc mở rộng các node ở các cấp độ trên nhiều lần.

Hình ảnh GIF minh họa: (Cần bổ sung hình ảnh GIF quá trình IDDFS trên 8-Puzzle)

Hình ảnh so sánh hiệu suất: (Cần bổ sung biểu đồ so sánh hiệu suất IDDFS với các thuật toán khác)

Nhận xét về hiệu suất trên 8-Puzzle: IDDFS thường hoạt động tốt trên 8-Puzzle, cân bằng giữa thời gian và bộ nhớ.


2.3. Các thuật toán Tìm kiếm có thông tin (Informed Search)

Các thuật toán tìm kiếm có thông tin sử dụng hàm heuristic (h(n)) để ước lượng chi phí từ trạng thái hiện tại (n) đến trạng thái đích. Điều này giúp hướng dẫn quá trình tìm kiếm về phía đích hiệu quả hơn.

Tìm kiếm tham lam (Greedy Best-First Search):

Nguyên lý: Luôn mở rộng node được đánh giá là "gần đích nhất" dựa trên hàm heuristic h(n). Sử dụng hàng đợi ưu tiên sắp xếp theo h(n).

Áp dụng cho 8-Puzzle: Sử dụng hàm heuristic (ví dụ: số ô sai vị trí hoặc tổng khoảng cách Manhattan của các ô so với vị trí đích) để ước lượng số bước còn lại để đạt đến đích.

Ưu điểm: Có thể tìm thấy lời giải rất nhanh nếu heuristic tốt.

Nhược điểm: Không đảm bảo tìm thấy lời giải tối ưu và có thể bị mắc kẹt trong các cực tiểu cục bộ hoặc đi theo đường đi không hiệu quả.

Hình ảnh GIF minh họa:

Nhận xét về hiệu suất trên 8-Puzzle: Greedy với heuristic tốt (như Manhattan) thường tìm thấy lời giải nhanh chóng. Tuy nhiên, lời giải tìm được có thể không phải là lời giải ngắn nhất.

Tìm kiếm A* (A* Search):

Nguyên lý: Mở rộng node có tổng chi phí ước lượng thấp nhất, được tính bằng f(n)=g(n)+h(n), trong đó g(n) là chi phí thực tế từ gốc đến n, và h(n) là chi phí ước lượng từ n đến đích. Sử dụng hàng đợi ưu tiên sắp xếp theo f(n).

Áp dụng cho 8-Puzzle: Kết hợp chi phí số bước đã đi với ước lượng số bước còn lại để đưa ra quyết định mở rộng node nào. Sử dụng heuristic Manhattan hoặc số ô sai vị trí.

Ưu điểm: Đảm bảo tìm thấy lời giải tối ưu nếu hàm heuristic là admissible (không bao giờ ước lượng quá cao chi phí thực tế đến đích) và consistent (thỏa mãn bất đẳng thức tam giác).

Nhược điểm: Có thể tốn nhiều bộ nhớ để lưu trữ các node đã mở rộng.

Hình ảnh GIF minh họa: (Cần bổ sung hình ảnh GIF quá trình A* trên 8-Puzzle)

Hình ảnh so sánh hiệu suất: (Cần bổ sung biểu đồ so sánh hiệu suất A* với các thuật toán khác)

Nhận xét về hiệu suất trên 8-Puzzle: A* với heuristic Manhattan là một trong những thuật toán hiệu quả nhất để giải 8-Puzzle, tìm được lời giải tối ưu với số node mở rộng tương đối ít.

Tìm kiếm IDA* (Iterative Deepening A* Search):

Nguyên lý: Kết hợp ý tưởng của IDDFS và A*. Thực hiện chuỗi các lần tìm kiếm theo chiều sâu có giới hạn chi phí (f-limit) tăng dần. Giới hạn f trong mỗi lần lặp là giá trị f(n) nhỏ nhất của các node bị cắt ở lần lặp trước.

Áp dụng cho 8-Puzzle: Tương tự IDDFS nhưng sử dụng f(n) làm tiêu chí giới hạn thay vì độ sâu.

Ưu điểm: Tìm thấy lời giải tối ưu và yêu cầu bộ nhớ ít như DFS. Thường là thuật toán tốt nhất khi bộ nhớ là yếu tố hạn chế.

Nhược điểm: Lặp lại việc mở rộng các node.

Hình ảnh GIF minh họa:

Nhận xét về hiệu suất trên 8-Puzzle: IDA* cũng rất hiệu quả trên 8-Puzzle và thường được ưa chuộng hơn A* khi cần tiết kiệm bộ nhớ.


2.4. Các thuật toán Tìm kiếm cục bộ (Local Search)

Các thuật toán tìm kiếm cục bộ hoạt động trên một trạng thái duy nhất (thay vì duy trì nhiều đường đi) và di chuyển từ trạng thái hiện tại sang trạng thái "tốt hơn" trong vùng lân cận của nó, dựa trên hàm mục tiêu hoặc heuristic. Chúng không lưu trữ đường đi và thường được sử dụng cho các bài toán tối ưu hóa hơn là tìm đường đi.

Leo đồi đơn giản (Simple Hill Climbing):

Nguyên lý: Từ trạng thái hiện tại, xem xét tất cả các trạng thái lân cận. Nếu có bất kỳ trạng thái lân cận nào tốt hơn trạng thái hiện tại (có giá trị heuristic thấp hơn), di chuyển đến trạng thái tốt hơn đầu tiên tìm thấy. Lặp lại cho đến khi không tìm thấy trạng thái lân cận nào tốt hơn.

Áp dụng cho 8-Puzzle: Bắt đầu từ một trạng thái ngẫu nhiên, liên tục di chuyển ô trống để giảm số ô sai vị trí hoặc tổng khoảng cách Manhattan.

Ưu điểm: Đơn giản, dễ cài đặt, yêu cầu bộ nhớ rất ít.

Nhược điểm: Dễ bị mắc kẹt trong các cực tiểu cục bộ (local optima) - trạng thái không phải đích nhưng không có lân cận nào tốt hơn. Không đảm bảo tìm thấy lời giải.

Hình ảnh GIF minh họa:

Nhận xét về hiệu suất trên 8-Puzzle: Hiếm khi tìm thấy lời giải tối ưu cho các bài toán 8-Puzzle phức tạp do không gian trạng thái có nhiều cực tiểu cục bộ.

Leo đồi dốc nhất (Steepest Ascent Hill Climbing):

Nguyên lý: Tương tự Simple Hill Climbing, nhưng thay vì di chuyển đến trạng thái tốt hơn đầu tiên, nó xem xét tất cả các lân cận và di chuyển đến trạng thái lân cận tốt nhất (giảm heuristic nhiều nhất).

Áp dụng cho 8-Puzzle: Luôn chọn nước đi làm giảm heuristic nhiều nhất.

Ưu điểm: Có thể hội tụ nhanh hơn Simple Hill Climbing.

Nhược điểm: Vẫn dễ bị mắc kẹt trong các cực tiểu cục bộ.

Hình ảnh GIF minh họa:


Nhận xét về hiệu suất trên 8-Puzzle: Tương tự Simple Hill Climbing, hiệu quả hạn chế trên 8-Puzzle.

Leo đồi ngẫu nhiên (Stochastic Hill Climbing):

Nguyên lý: Chọn ngẫu nhiên một trạng thái lân cận tốt hơn để di chuyển, thay vì chọn trạng thái tốt nhất hoặc trạng thái đầu tiên.

Áp dụng cho 8-Puzzle: Chọn ngẫu nhiên một nước đi làm giảm heuristic.

Ưu điểm: Có thể thoát khỏi một số cực tiểu cục bộ hẹp.

Nhược điểm: Vẫn không đảm bảo tìm thấy lời giải tối ưu và có thể mất nhiều thời gian hơn để hội tụ.

Hình ảnh GIF minh họa: (Cần bổ sung hình ảnh GIF quá trình Stochastic Hill Climbing trên 8-Puzzle)

Hình ảnh so sánh hiệu suất: (Cần bổ sung biểu đồ so sánh hiệu suất Stochastic Hill Climbing với các thuật toán khác)

Nhận xét về hiệu suất trên 8-Puzzle: Hiệu quả tương tự các biến thể Hill Climbing khác trên 8-Puzzle.

Mô phỏng luyện kim (Simulated Annealing):

Nguyên lý: Thuật toán tìm kiếm cục bộ probabilistic lấy cảm hứng từ quá trình làm nguội kim loại. Nó chấp nhận di chuyển đến trạng thái "tệ hơn" (tăng heuristic) với một xác suất giảm dần theo thời gian (nhiệt độ giảm). Điều này giúp thoát khỏi các cực tiểu cục bộ.

Áp dụng cho 8-Puzzle: Sử dụng heuristic (ví dụ: Manhattan) và một lịch trình nhiệt độ giảm dần để khám phá không gian trạng thái.

Ưu điểm: Có khả năng thoát khỏi cực tiểu cục bộ và tìm thấy lời giải gần tối ưu hoặc tối ưu (nếu lịch trình nhiệt độ đủ chậm).

Nhược điểm: Việc lựa chọn lịch trình nhiệt độ rất quan trọng và có thể khó khăn.

Hình ảnh GIF minh họa:

Nhận xét về hiệu suất trên 8-Puzzle: Có thể tìm thấy lời giải cho 8-Puzzle, nhưng không đảm bảo tính tối ưu. Hiệu quả phụ thuộc nhiều vào các tham số (nhiệt độ ban đầu, tốc độ giảm nhiệt).

Tìm kiếm chùm (Beam Search):

Nguyên lý: Duy trì một tập hợp cố định gồm k trạng thái tốt nhất (được gọi là chùm - beam) ở mỗi bước. Từ các trạng thái trong chùm hiện tại, tạo ra tất cả các trạng thái lân cận, sau đó chọn k trạng thái tốt nhất từ tập hợp lân cận này để tạo thành chùm tiếp theo.

Áp dụng cho 8-Puzzle: Ở mỗi bước, tạo ra tất cả các trạng thái con từ chùm hiện tại và chỉ giữ lại k trạng thái tốt nhất theo heuristic.

Ưu điểm: Yêu cầu bộ nhớ giới hạn (theo kích thước chùm k). Có thể tìm thấy lời giải nhanh hơn các thuật toán mù nếu heuristic tốt.

Nhược điểm: Không đảm bảo tìm thấy lời giải tối ưu (trừ khi k đủ lớn, khi đó nó trở thành BFS). Có thể bỏ sót lời giải nếu nó nằm ngoài chùm.

Hình ảnh GIF minh họa: 

Nhận xét về hiệu suất trên 8-Puzzle: Hiệu quả phụ thuộc vào kích thước chùm. Kích thước chùm lớn hơn tăng khả năng tìm thấy lời giải tốt nhưng tốn bộ nhớ hơn.

Thuật toán Di truyền (Genetic Algorithm):

Nguyên lý: Thuật toán tối ưu hóa dựa trên cơ chế tiến hóa tự nhiên. Duy trì một quần thể các "cá thể" (ở đây là các chuỗi hành động tiềm năng). Các cá thể tốt hơn (có fitness cao hơn, tức là đưa đến trạng thái gần đích hơn) có cơ hội được chọn để "sinh sản" (kết hợp - crossover) và "đột biến" (mutation) để tạo ra thế hệ mới. Quá trình lặp lại qua nhiều thế hệ.

Áp dụng cho 8-Puzzle: Mỗi cá thể là một chuỗi các bước di chuyển. Fitness được đánh giá dựa trên mức độ gần đích của trạng thái cuối cùng sau khi thực hiện chuỗi hành động đó.

Ưu điểm: Có khả năng tìm kiếm trong không gian rộng và thoát khỏi các cực tiểu cục bộ.

Nhược điểm: Không đảm bảo tìm thấy lời giải tối ưu hoặc thậm chí là lời giải. Việc lựa chọn các tham số (kích thước quần thể, tỷ lệ đột biến, v.v.) rất quan trọng.

Hình ảnh GIF minh họa:

Nhận xét về hiệu suất trên 8-Puzzle: Genetic Algorithm có thể tìm thấy lời giải cho 8-Puzzle, nhưng thường chậm hơn các thuật toán tìm kiếm có thông tin như A* và không đảm bảo tính tối ưu.


2.5. Các thuật toán Tìm kiếm dựa trên ràng buộc (Constraint-Based Search)

Tìm kiếm dựa trên ràng buộc (Constraint Satisfaction Problem - CSP) là một kỹ thuật giải quyết vấn đề trong đó lời giải phải thỏa mãn một tập hợp các ràng buộc. Mặc dù 8-Puzzle có thể được mô hình hóa như một bài toán CSP (ví dụ: mỗi ô là một biến có miền giá trị từ 0-8, và các ràng buộc là vị trí tương đối của các ô), việc giải nó bằng các thuật toán CSP truyền thống (như Backtracking Search, Forward Checking, AC3) thường không hiệu quả bằng các thuật toán tìm kiếm trạng thái, đặc biệt là khi tìm đường đi ngắn nhất.

Tìm kiếm quay lui (Backtracking Search):

Nguyên lý: Duyệt cây tìm kiếm theo chiều sâu, gán giá trị cho từng biến một. Nếu tại một bước nào đó, việc gán giá trị vi phạm một ràng buộc, thuật toán quay lui về biến trước đó và thử giá trị khác.

Áp dụng cho 8-Puzzle (mô hình hóa CSP): Gán vị trí cho từng ô số. Nếu việc gán một ô vi phạm ràng buộc (ví dụ: hai ô cùng vị trí), quay lui.

Ưu điểm: Đơn giản.

Nhược điểm: Có thể rất chậm do phải khám phá nhiều nhánh không dẫn đến lời giải.

Hình ảnh GIF minh họa:

Nhận xét về hiệu suất trên 8-Puzzle: Backtracking Search trực tiếp trên mô hình CSP của 8-Puzzle thường không hiệu quả bằng các thuật toán tìm kiếm trạng thái.

Kiểm tra tiến (Forward Checking):

Nguyên lý: Một cải tiến của Backtracking Search. Khi gán giá trị cho một biến, thuật toán kiểm tra các ràng buộc liên quan đến biến đó và thu hẹp miền giá trị của các biến chưa được gán mà bị ảnh hưởng bởi ràng buộc đó. Nếu miền giá trị của bất kỳ biến nào trở nên rỗng, thuật toán quay lui ngay lập tức.

Áp dụng cho 8-Puzzle (mô hình hóa CSP): Khi đặt một ô vào vị trí, kiểm tra xem vị trí đó có hợp lệ với các ô khác chưa được đặt hay không và loại bỏ các vị trí không hợp lệ cho các ô đó.

Ưu điểm: Phát hiện sớm hơn các trường hợp không khả thi so với Backtracking Search.

Nhược điểm: Vẫn có thể tốn thời gian với các bài toán lớn.

Hình ảnh GIF minh họa:

Nhận xét về hiệu suất trên 8-Puzzle: Hiệu quả hơn Backtracking Search nhưng vẫn kém hơn các thuật toán tìm kiếm trạng thái có thông tin.

Lan truyền ràng buộc (Constraint Propagation - AC3):

Nguyên lý: Thuật toán AC3 (Arc Consistency 3) đảm bảo tính nhất quán cung (arc consistency) giữa các cặp biến. Đối với mỗi cung (Xi​,Xj​), nó loại bỏ các giá trị khỏi miền giá trị của Xi​ mà không có giá trị tương ứng trong miền giá trị của Xj​ thỏa mãn ràng buộc giữa Xi​ và Xj​. Quá trình lặp lại cho đến khi không có miền giá trị nào thay đổi nữa.

Áp dụng cho 8-Puzzle (mô hình hóa CSP): Áp dụng các ràng buộc về vị trí tương đối giữa các ô để thu hẹp miền giá trị các vị trí khả thi cho từng ô.

Ưu điểm: Có thể thu hẹp đáng kể không gian tìm kiếm trước khi thực hiện tìm kiếm.

Nhược điểm: Không đảm bảo tìm thấy lời giải; thường cần kết hợp với tìm kiếm (như Backtracking) để tìm lời giải cụ thể.

Hình ảnh GIF minh họa:

Nhận xét về hiệu suất trên 8-Puzzle: AC3 được sử dụng để tiền xử lý CSP 8-Puzzle, giúp giảm kích thước miền giá trị, nhưng không phải là thuật toán giải toàn diện.


2.6. Các thuật toán Tìm kiếm phức tạp (Complex Search)

Các thuật toán này giải quyết các bài toán tìm kiếm trong các môi trường phức tạp hơn, ví dụ như môi trường không chắc chắn hoặc có thông tin không đầy đủ.

Tìm kiếm trạng thái niềm tin (Belief State Search):

Nguyên lý: Trong môi trường không chắc chắn về trạng thái ban đầu, thuật toán hoạt động trên "trạng thái niềm tin" (belief state), là một tập hợp các trạng thái vật lý có thể xảy ra. Một hành động được thực hiện sẽ dẫn đến một trạng thái niềm tin mới, là tập hợp các trạng thái vật lý có thể đạt được từ tất cả các trạng thái trong trạng thái niềm tin cũ sau khi thực hiện hành động đó.

Áp dụng cho 8-Puzzle: Nếu không biết chính xác cấu hình ban đầu của bàn cờ, nhưng biết nó là một trong một vài cấu hình có thể có, thuật toán sẽ tìm một chuỗi hành động chung để giải quyết tất cả các trường hợp đó.

Ưu điểm: Có thể giải quyết bài toán trong môi trường không chắc chắn về trạng thái ban đầu.

Nhược điểm: Không gian trạng thái niềm tin có thể rất lớn (tập hợp các tập hợp trạng thái vật lý).

Hình ảnh GIF minh họa:

Nhận xét về hiệu suất trên 8-Puzzle: Hiệu quả phụ thuộc vào số lượng trạng thái ban đầu có thể có. Số lượng trạng thái ban đầu càng lớn, không gian tìm kiếm trạng thái niềm tin càng phức tạp.

Tìm kiếm AND-OR (AND-OR Search):

Nguyên lý: Được sử dụng để giải quyết các bài toán trong môi trường có kết quả hành động không chắc chắn (ví dụ: trò chơi, lập kế hoạch với các sự kiện ngẫu nhiên). Cây tìm kiếm bao gồm các node OR (đại diện cho các lựa chọn hành động của tác nhân) và các node AND (đại diện cho các kết quả có thể xảy ra sau hành động do môi trường hoặc đối thủ tạo ra).

Áp dụng cho 8-Puzzle: Có thể mô hình hóa nếu có yếu tố không chắc chắn trong việc di chuyển (ví dụ: đôi khi di chuyển sang trái lại dẫn đến di chuyển xuống).

Ưu điểm: Phù hợp với các bài toán có yếu tố không chắc chắn hoặc đối kháng.

Nhược điểm: Phức tạp để xây dựng cây AND-OR và thực hiện tìm kiếm.

Hình ảnh GIF minh họa: 

Nhận xét về hiệu suất trên 8-Puzzle: Ít phổ biến cho bài toán 8-Puzzle tiêu chuẩn do tính xác định của nó, nhưng có thể áp dụng cho các biến thể có thêm yếu tố ngẫu nhiên.

Tìm kiếm với quan sát một phần (Search with Partial Observation):

Nguyên lý: Tương tự Belief State Search, nhưng tác nhân chỉ có thể quan sát một phần trạng thái vật lý sau mỗi hành động. Thông tin quan sát được sử dụng để cập nhật trạng thái niềm tin.

Áp dụng cho 8-Puzzle: Nếu sau khi di chuyển, tác nhân chỉ biết vị trí của một vài ô nhất định, nó cần suy luận về các cấu hình bàn cờ có thể xảy ra dựa trên quan sát đó.

Ưu điểm: Giải quyết bài toán trong môi trường có thông tin không đầy đủ sau hành động.

Nhược điểm: Việc cập nhật trạng thái niềm tin dựa trên quan sát một phần có thể phức tạp.

Hình ảnh GIF minh họa:

Nhận xét về hiệu suất trên 8-Puzzle: Phức tạp hơn Belief State Search do cần xử lý thông tin quan sát không đầy đủ.


2.7. Thuật toán Học tăng cường (Reinforcement Learning Search)

Học tăng cường (RL) là một lĩnh vực của AI nơi tác nhân học cách hành động trong một môi trường để tối đa hóa phần thưởng tích lũy. Thay vì tìm kiếm đường đi trong không gian trạng thái đã biết, tác nhân RL học thông qua thử và sai.

Học Q-Learning (Q-Learning):

Nguyên lý: Một thuật toán RL không mô hình (model-free). Tác nhân học một hàm giá trị Q (Q(s,a)), ước lượng giá trị kỳ vọng của việc thực hiện hành động a trong trạng thái s và sau đó tuân theo chính sách tối ưu. Tác nhân cập nhật giá trị Q dựa trên phần thưởng nhận được và giá trị Q của trạng thái tiếp theo.

Áp dụng cho 8-Puzzle: Tác nhân (chương trình AI) học cách di chuyển các ô bằng cách thử các hành động khác nhau từ các trạng thái khác nhau. Nó nhận phần thưởng (ví dụ: dương khi đạt đích, âm khi di chuyển sai hoặc theo khoảng cách Manhattan giảm) và cập nhật bảng Q để biết nước đi nào là tốt nhất trong mỗi trạng thái.

Ưu điểm: Có thể học được chính sách tối ưu mà không cần biết trước mô hình chuyển đổi của môi trường.

Nhược điểm: Có thể yêu cầu rất nhiều lần thử (episodes) để hội tụ, đặc biệt với không gian trạng thái lớn. Việc lựa chọn các tham số (tốc độ học α, hệ số chiết khấu γ, tham số khám phá ϵ) rất quan trọng.

Hình ảnh GIF minh họa

Nhận xét về hiệu suất trên 8-Puzzle: Q-Learning có thể giải 8-Puzzle sau quá trình huấn luyện đủ dài. Hiệu quả thực thi (thời gian tìm lời giải sau khi huấn luyện) thường nhanh, nhưng quá trình huấn luyện có thể tốn thời gian.


4. Kết luận

Đồ án đã hoàn thành các mục tiêu đề ra, triển khai thành công một loạt các thuật toán tìm kiếm kinh điển và hiện đại để giải quyết bài toán 8-Puzzle. Các kết quả đạt được bao gồm:
Xây dựng được giao diện người dùng trực quan cho phép tương tác với bài toán 8-Puzzle và các thuật toán.

Triển khai thành công các thuật toán tìm kiếm không có thông tin (BFS, DFS, UCS, IDDFS), có thông tin (Greedy, A*, IDA*), cục bộ (Hill Climbing, Simulated Annealing, Beam Search, Genetic Algorithm), dựa trên ràng buộc (Backtracking, Forward Checking, AC3), tìm kiếm phức tạp (Belief State, AND-OR, Partial Observation) và học tăng cường (Q-Learning).

Cung cấp khả năng trực quan hóa quá trình giải của thuật toán và so sánh hiệu suất giữa chúng.

Hiểu rõ hơn về nguyên lý hoạt động, ưu nhược điểm và phạm vi áp dụng của từng loại thuật toán tìm kiếm thông qua việc giải quyết một bài toán cụ thể.

Project này không chỉ là sản phẩm cuối kỳ mà còn là cơ hội quý báu để nhóm củng cố kiến thức lý thuyết và rèn luyện kỹ năng lập trình trong lĩnh vực Trí tuệ Nhân tạo. Mặc dù còn một số hạn chế (ví dụ: cần tối ưu hóa hiệu suất cho các bài toán rất khó, giao diện có thể cải thiện thêm), nhưng những kết quả đạt được là nền tảng quan trọng cho việc tiếp tục tìm hiểu và làm việc với các bài toán AI phức tạp hơn trong tương lai.
