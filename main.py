import numpy as np
import sympy as sp
import tkinter as tk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk


class Placa:
    def __init__(self, a, b, espessura, E, v):
        self.a = a
        self.b = b
        self.espessura = espessura
        self.E = E
        self.v = v
        self.D = (E * (espessura**3)) / (12 * (1 - v**2))
        self.x, self.y = sp.symbols('x y')

    def definir_carregamento(self, tipo, p, x1=None, y1=None, centro_x=None, centro_y=None, largura_patch=None, comprimento_patch=None):
        if tipo == 'Distribuída':
            p_eq = sp.sympify(p)
        elif tipo == 'Pontual':
            if x1 is None or y1 is None:
                raise ValueError("Coordenadas x1 e y1 são necessárias para carga pontual")
            p_eq = sp.sympify(p) * sp.DiracDelta(self.x - x1) * sp.DiracDelta(self.y - y1)
        elif tipo == 'Patch':
            if centro_x is None or centro_y is None or largura_patch is None or comprimento_patch is None:
                raise ValueError("Centro e dimensões do patch são necessários para carga patch")
            p_eq = sp.sympify(p) * sp.Piecewise(
                (1, (self.x >= centro_x - largura_patch/2) & (self.x <= centro_x + largura_patch/2) &
                    (self.y >= centro_y - comprimento_patch/2) & (self.y <= centro_y + comprimento_patch/2)),
                (0, True)
            )
        else:
            raise ValueError("Tipo de carregamento desconhecido")
        return p_eq

    def p_mn(self, m, n, p):
        integral = sp.integrate(sp.integrate(p * sp.sin(m * sp.pi * self.x / self.a) * sp.sin(n * sp.pi * self.y / self.b), (self.x, 0, self.a)), (self.y, 0, self.b))
        return (4 / (self.a * self.b)) * integral

    def a_mn(self, m, n, p):
        self.p_mn_val = self.p_mn(m, n, p)
        return self.p_mn_val / (self.D * ((m * sp.pi / self.a)**2 + (n * sp.pi / self.b)**2)**2)

    def deflexao_serie(self, m_max, n_max, p):
        w_serie = 0
        for m in range(1, m_max + 1):
            for n in range(1, n_max + 1):
                a_mn_val = self.a_mn(m, n, p)
                w_serie += a_mn_val * sp.sin(m * sp.pi * self.x / self.a) * sp.sin(n * sp.pi * self.y / self.b)
        return w_serie

    def max_deflexao(self, w_serie):
        w_max = sp.lambdify((self.x, self.y), w_serie, 'numpy')
        x_vals = np.linspace(0, self.a, 100)
        y_vals = np.linspace(0, self.b, 100)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = w_max(X, Y)
        return np.max(Z)

    def momentos(self, w_serie):
        w_func = sp.lambdify((self.x, self.y), w_serie, 'numpy')
        x_vals = np.linspace(0, self.a, 100)
        y_vals = np.linspace(0, self.b, 100)
        X, Y = np.meshgrid(x_vals, y_vals)
        W = w_func(X, Y)

        dx = x_vals[1] - x_vals[0]
        dy = y_vals[1] - y_vals[0]

        Mx = np.gradient(np.gradient(W, axis=0), axis=0) * (self.D / (dx ** 2))
        My = np.gradient(np.gradient(W, axis=1), axis=1) * (self.D / (dy ** 2))
        Mxy = np.gradient(np.gradient(W, axis=0), axis=1) * (self.D / (dx * dy))

        Qx = np.gradient(Mx, axis=1)
        Qy = np.gradient(My, axis=0)

        return X, Y, W, Mx, My, Mxy, Qx, Qy
        
    def convergencia(self, p, tolerancia=0.05):
        deflexao_max_anterior = None
        m = n = 1

        while True:
            w_serie = self.deflexao_serie(m, n, p)
            deflexao_max_atual = self.max_deflexao(w_serie)
            
            if deflexao_max_anterior is not None:
                if (abs(deflexao_max_atual - deflexao_max_anterior) / abs(deflexao_max_anterior) < tolerancia) and (m > 1 and n > 1):
                    break
            
            if m == 1 and n == 1:
                break
            
            deflexao_max_anterior = deflexao_max_atual
            m += 1
            n += 1

        return m, n, w_serie


    def F_resultante(self, p):
        integral = sp.integrate(sp.integrate(p, (self.x, 0, self.a)), (self.y, 0, self.b))
        return integral

    def torcao_cantos(self, w_serie):
        Mxy_expr = self.D * sp.diff(w_serie, self.x, self.y)
        Mxy_func = sp.lambdify((self.x, self.y), Mxy_expr, 'numpy')
        cantos = [(0, 0), (0, self.b), (self.a, 0), (self.a, self.b)]
        return [Mxy_func(c[0], c[1]) for c in cantos]

    def reacoes(self, w_serie):
        momentos_torcao_cantos = self.torcao_cantos(w_serie)
        R_cantos = [2 * Mxy_canto for Mxy_canto in momentos_torcao_cantos]
        
        return R_cantos

class InterfaceGrafica:
    def __init__(self, root):
        self.root = root
        self.root.title("Análise de Placa")

        self.root.attributes('-fullscreen', True)  # Ativa o modo de tela cheia
        self.root.bind("<Escape>", self.toggle_fullscreen)

        # Frame para inputs e resultados
        self.frame_main = tk.Frame(self.root)
        self.frame_main.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.frame_inputs = tk.Frame(self.frame_main)
        self.frame_inputs.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        self.frame_resultados = tk.Frame(self.frame_main)
        self.frame_resultados.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Frame para gráficos
        self.frame_graficos = tk.Frame(self.root)
        self.frame_graficos.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.figura = plt.Figure(figsize=(15, 12), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figura, master=self.frame_graficos)

        # Barra de ferramentas
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.frame_graficos)
        self.toolbar.update()

        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Labels e Entradas para as propriedades da placa
        tk.Label(self.frame_inputs, text="Largura (x): ").grid(row=0, column=0, sticky='w', padx=5, pady=5)
        tk.Label(self.frame_inputs, text="Comprimento (y): ").grid(row=1, column=0, sticky='w', padx=5, pady=5)
        tk.Label(self.frame_inputs, text="Espessura (z): ").grid(row=2, column=0, sticky='w', padx=5, pady=5)
        tk.Label(self.frame_inputs, text="Módulo de Young:").grid(row=3, column=0, sticky='w', padx=5, pady=5)
        tk.Label(self.frame_inputs, text="Poisson:").grid(row=4, column=0, sticky='w', padx=5, pady=5)
        tk.Label(self.frame_inputs, text="Carga:").grid(row=5, column=0, sticky='w', padx=5, pady=5)

        self.entry_a = tk.Entry(self.frame_inputs)
        self.entry_b = tk.Entry(self.frame_inputs)
        self.entry_espessura = tk.Entry(self.frame_inputs)
        self.entry_E = tk.Entry(self.frame_inputs)
        self.entry_v = tk.Entry(self.frame_inputs)
        self.entry_funcao = tk.Entry(self.frame_inputs)

        self.entry_a.grid(row=0, column=1, padx=5, pady=5)
        self.entry_b.grid(row=1, column=1, padx=5, pady=5)
        self.entry_espessura.grid(row=2, column=1, padx=5, pady=5)
        self.entry_E.grid(row=3, column=1, padx=5, pady=5)
        self.entry_v.grid(row=4, column=1, padx=5, pady=5)
        self.entry_funcao.grid(row=5, column=1, padx=5, pady=5)

        # Tipo de Carregamento
        tk.Label(self.frame_inputs, text="Tipo de Carregamento:").grid(row=6, column=0, sticky='w', padx=5, pady=5)
        self.tipo_carregamento_var = tk.StringVar(value="Distribuída")
        self.menu_tipo_carregamento = tk.OptionMenu(self.frame_inputs, self.tipo_carregamento_var,"Distribuída", "Pontual", "Patch",command=self.atualizar_campos)
        self.menu_tipo_carregamento.grid(row=6, column=1, padx=5, pady=5)

        # Labels e Entradas para x1 e y1 (para carga pontual)
        self.label_x1 = tk.Label(self.frame_inputs, text= "X:")
        self.label_y1 = tk.Label(self.frame_inputs, text= "Y:")
        self.x1 = tk.StringVar()
        self.y1 = tk.StringVar()
        self.entry_x1 = tk.Entry(self.frame_inputs,textvariable=self.x1)
        self.entry_y1 = tk.Entry(self.frame_inputs,textvariable=self.y1)

        # Labels e Entradas para centro da região, largura e comprimento (para carga distribuída)
        self.label_centro_x = tk.Label(self.frame_inputs, text="X:")
        self.label_centro_y = tk.Label(self.frame_inputs, text="Y:")
        self.label_largura_patch = tk.Label(self.frame_inputs, text="Largura:")
        self.label_comprimento_patch = tk.Label(self.frame_inputs, text="Comprimento:")

        self.entry_centro_x = tk.Entry(self.frame_inputs)
        self.entry_centro_y = tk.Entry(self.frame_inputs)
        self.entry_largura_patch = tk.Entry(self.frame_inputs)
        self.entry_comprimento_patch = tk.Entry(self.frame_inputs)

        tk.Button(self.frame_inputs, text="Calcular", command=self.on_calcular).grid(row=11, column=0, columnspan=3, pady=5, padx=5)
        tk.Button(self.frame_inputs, text="Resetar", command=self.resetar).grid(row=11, column=1, columnspan=3, pady=5, padx=5)

        self.text_area = tk.Text(self.frame_resultados, height=15, width=30)
        self.text_area.pack(fill=tk.BOTH, expand=True)

        self.desenhar_plano_branco()

    def atualizar_campos(self, tipo_carregamento):
        if tipo_carregamento == "Pontual":
            self.exibir_campos_pontual()
            self.ocultar_campos_distribuida()
        elif tipo_carregamento == "Patch":
            self.ocultar_campos_pontual()
            self.exibir_campos_distribuida()
        else:
            self.ocultar_campos_pontual()
            self.ocultar_campos_distribuida()

    def exibir_campos_pontual(self):
        self.label_x1.grid(row=7, column=0, sticky='w', padx=5, pady=5)
        self.label_y1.grid(row=8, column=0, sticky='w', padx=5, pady=5)
        self.entry_x1.grid(row=7, column=1, padx=5, pady=5)
        self.entry_y1.grid(row=8, column=1, padx=5, pady=5)

    def ocultar_campos_pontual(self):
        self.label_x1.grid_forget()
        self.label_y1.grid_forget()
        self.entry_x1.grid_forget()
        self.entry_y1.grid_forget()

    def exibir_campos_distribuida(self):
        self.label_centro_x.grid(row=7, column=0, sticky='w', padx=5, pady=5)
        self.label_centro_y.grid(row=8, column=0, sticky='w', padx=5, pady=5)
        self.label_largura_patch.grid(row=9, column=0, sticky='w', padx=5, pady=5)
        self.label_comprimento_patch.grid(row=10, column=0, sticky='w', padx=5, pady=5)

        self.entry_centro_x.grid(row=7, column=1, padx=5, pady=5)
        self.entry_centro_y.grid(row=8, column=1, padx=5, pady=5)
        self.entry_largura_patch.grid(row=9, column=1, padx=5, pady=5)
        self.entry_comprimento_patch.grid(row=10, column=1, padx=5, pady=5)

    def ocultar_campos_distribuida(self):
        self.label_centro_x.grid_forget()
        self.label_centro_y.grid_forget()
        self.label_largura_patch.grid_forget()
        self.label_comprimento_patch.grid_forget()
        self.entry_centro_x.grid_forget()
        self.entry_centro_y.grid_forget()
        self.entry_largura_patch.grid_forget()
        self.entry_comprimento_patch.grid_forget()

    def resetar(self):
        self.entry_a.delete(0, tk.END)
        self.entry_b.delete(0, tk.END)
        self.entry_espessura.delete(0, tk.END)
        self.entry_E.delete(0, tk.END)
        self.entry_v.delete(0, tk.END)
        self.entry_funcao.delete(0, tk.END)
        self.entry_x1.delete(0, tk.END)
        self.entry_y1.delete(0, tk.END)
        self.entry_centro_x.delete(0, tk.END)
        self.entry_centro_y.delete(0, tk.END)
        self.entry_largura_patch.delete(0, tk.END)
        self.entry_comprimento_patch.delete(0, tk.END)

        self.tipo_carregamento_var.set("Pontual")
        self.ocultar_campos_distribuida()
        self.exibir_campos_pontual()

        self.desenhar_plano_branco()
    def desenhar_plano_branco(self):
        self.figura.clear()

        ax1 = self.figura.add_subplot(231, projection='3d')
        ax2 = self.figura.add_subplot(232, projection='3d')
        ax3 = self.figura.add_subplot(233, projection='3d')
        ax4 = self.figura.add_subplot(234, projection='3d')
        ax5 = self.figura.add_subplot(235, projection='3d')
        ax6 = self.figura.add_subplot(236, projection='3d')

        a = float(self.entry_a.get() or 1)
        b = float(self.entry_b.get() or 1)

        X, Y = np.meshgrid([0, a], [0, b])
        Z = np.zeros_like(X)

        # Desenhar a placa em todos os gráficos
        def plotar_placa(ax):
            ax.plot_surface(X, Y, Z, color='black', alpha=0.3)

        plotar_placa(ax1)
        plotar_placa(ax2)
        plotar_placa(ax3)
        plotar_placa(ax4)
        plotar_placa(ax5)
        plotar_placa(ax6)

        ax1.set_title('Deflexão w')
        ax2.set_title('Momento M_x')
        ax3.set_title('Momento M_y')
        ax4.set_title('Momento M_{xy}')
        ax5.set_title('Força Q_x')
        ax6.set_title('Força Q_y')

        self.canvas.draw()

    def desenhar_graficos(self, X, Y, W, Mx, My, Mxy, Qx, Qy):
        self.figura.clear()

        ax1 = self.figura.add_subplot(231, projection='3d')
        ax2 = self.figura.add_subplot(232, projection='3d')
        ax3 = self.figura.add_subplot(233, projection='3d')
        ax4 = self.figura.add_subplot(234, projection='3d')
        ax5 = self.figura.add_subplot(235, projection='3d')
        ax6 = self.figura.add_subplot(236, projection='3d')

        def plotar_placa(ax):
            ax.plot_surface(X, Y, np.zeros_like(X), color='black', alpha=0.3)

        def plotar_superficie(ax, Z, cmap):
            surf = ax.plot_surface(X, Y, Z, cmap=cmap, edgecolor='none')
            return surf

        plotar_placa(ax1)
        plotar_placa(ax2)
        plotar_placa(ax3)
        plotar_placa(ax4)
        plotar_placa(ax5)
        plotar_placa(ax6)

        surf1 = plotar_superficie(ax1, W, 'viridis')
        surf2 = plotar_superficie(ax2, Mx, 'plasma')
        surf3 = plotar_superficie(ax3, My, 'inferno')
        surf4 = plotar_superficie(ax4, Mxy, 'magma')
        surf5 = plotar_superficie(ax5, Qx, 'cividis')
        surf6 = plotar_superficie(ax6, Qy, 'coolwarm')

        ax1.set_title('Deflexão w')
        ax2.set_title('Momento M_x')
        ax3.set_title('Momento M_y')
        ax4.set_title('Momento M_{xy}')
        ax5.set_title('Força Q_x')
        ax6.set_title('Força Q_y')

        self.canvas.draw()

    def on_calcular(self):
        try:
            a = float(self.entry_a.get())
            b = float(self.entry_b.get())
            espessura = float(self.entry_espessura.get())
            E = float(self.entry_E.get())
            v = float(self.entry_v.get())
            funcao_p = self.entry_funcao.get()

            placa = Placa(a, b, espessura, E, v)
            
            if self.tipo_carregamento_var.get() == "Pontual":
                p_eq = placa.definir_carregamento(self.tipo_carregamento_var.get(),funcao_p,x1=float(self.x1.get()),y1=float(self.y1.get()))
            elif self.tipo_carregamento_var.get() == "Patch":
                p_eq = placa.definir_carregamento(self.tipo_carregamento_var.get(),funcao_p,centro_x=float(self.entry_centro_x.get()),
                                                    centro_y=float(self.entry_centro_y.get()), largura_patch=float(self.entry_largura_patch.get()),
                                                    comprimento_patch=float(self.entry_comprimento_patch.get()))
            else:
                p_eq = placa.definir_carregamento(self.tipo_carregamento_var.get(),funcao_p)

            m_max, n_max, w_serie = placa.convergencia(p_eq)

            X, Y, W, Mx, My, Mxy, Qx, Qy = placa.momentos(w_serie)

            self.desenhar_graficos(X, Y, W, Mx, My, Mxy, Qx, Qy)

            max_deflexao = placa.max_deflexao(w_serie)
            reacoes_pontuais = placa.reacoes(w_serie)

            resultado_texto = f"""Resultados:

    Pmn: {placa.p_mn_val}

    Deflexão: {w_serie}

    Convergência: m,n = {m_max},{n_max}

    Deflexão Máxima: {max_deflexao:.4f} 
            
    {"Canto":<15} {"Reação":<15}
    {"(0,0)":<15} {reacoes_pontuais[0]:<15.4f}
    {f"(0,{b})":<15} {reacoes_pontuais[1]:<15.4f}
    {f"({a},0)":<15} {reacoes_pontuais[2]:<15.4f}
    {f"({a},{b})":<15} {reacoes_pontuais[3]:<15.4f}
    """
            self.text_area.delete(1.0, tk.END)
            self.text_area.insert(tk.END, resultado_texto)
        except Exception as e:
            messagebox.showerror("Erro", str(e))

    def toggle_fullscreen(self, event=None):
        self.root.attributes('-fullscreen', not self.root.attributes('-fullscreen'))
        return "break"


if __name__ == "__main__":
    root = tk.Tk()
    app = InterfaceGrafica(root)
    root.mainloop()